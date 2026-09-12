"""Direct TensorRT runner for the YOLO26-seg person detector.

Ultralytics' wrapper costs almost nothing on a desktop GPU but dominates on a
Jetson: measured on an AGX Orin, the yolo26n-seg engine itself runs in 3.05 ms
while ultralytics reports 12.32 ms for the same engine -- the extra ~9 ms is
host-side work (numpy letterbox, H2D copy, tensor reshuffling, mask assembly on
the CPU).  On an RTX 4070 the same wrapper adds only 0.2 ms, which is why this
never showed up before.

So do the whole thing on the GPU: upload the frame once, letterbox and normalise
with torch, run the engine, and assemble the instance mask from the prototype
bank without a round trip.  Only the final person mask comes back to the host.

YOLO26 is NMS-free, so `output0` is already the final detection list:
    output0 (1, N, 38) = [x1, y1, x2, y2, conf, cls, 32 mask coefficients]
    output1 (1, 32, 160, 160) = mask prototypes over the letterboxed canvas
"""
from pathlib import Path

import numpy as np
import torch


class YoloSegTRT:
    """Person box + full-resolution bool mask, computed entirely on the GPU."""

    def __init__(self, path, conf=0.4, person_class=0, device="cuda",
                 classes=None):
        import tensorrt as trt
        self.conf = float(conf)
        self.person_class = int(person_class)
        # None means "every class the engine knows".  detect_all() uses this;
        # __call__ keeps its own single-class filter so the pose demos are
        # unchanged.
        self.classes = None if classes is None else sorted(int(c) for c in classes)
        self.device = torch.device(device)
        blob = self._strip_ultralytics_header(Path(path).read_bytes())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()

        dmap = {}
        for name, tt in (("float32", torch.float32), ("float16", torch.float16),
                         ("int32", torch.int32), ("int64", torch.int64),
                         ("int8", torch.int8), ("bool", torch.bool)):
            if hasattr(trt, name):
                dmap[getattr(trt, name)] = tt

        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            buf = torch.zeros(shape, device=self.device,
                              dtype=dmap[self.engine.get_tensor_dtype(n)])
            self.context.set_tensor_address(n, buf.data_ptr())
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inputs[n] = buf
            else:
                self.outputs[n] = buf
        if len(self.inputs) != 1:
            raise RuntimeError(f"expected one input, got {list(self.inputs)}")
        self.in_name, self.in_buf = next(iter(self.inputs.items()))
        self.net_h, self.net_w = self.in_buf.shape[-2:]

        # Detections come out as (1, N, 38); prototypes as (1, 32, H/4, W/4).
        self.det_name = min(self.outputs, key=lambda n: self.outputs[n].numel())
        self.proto_name = max(self.outputs, key=lambda n: self.outputs[n].numel())
        self.stream = torch.cuda.Stream()
        self._src_hw = None

    @staticmethod
    def _strip_ultralytics_header(blob):
        """Ultralytics prefixes engines with a 4-byte length + JSON metadata."""
        n = int.from_bytes(blob[:4], "little")
        if 0 < n < 100_000 and blob[4:5] == b"{":
            return blob[4 + n:]
        return blob

    def _letterbox_params(self, h, w):
        if self._src_hw != (h, w):
            r = min(self.net_h / h, self.net_w / w)
            new_h, new_w = round(h * r), round(w * r)
            self._src_hw = (h, w)
            self._lb = (r, new_h, new_w,
                        (self.net_h - new_h) // 2, (self.net_w - new_w) // 2)
        return self._lb

    def __call__(self, rgb, all_people=False):
        """rgb: HxWx3 uint8 (RGB).

        Returns (box_xyxy, mask HxW bool) for the largest person, or None.
        With all_people=True, returns a list of (box, mask, conf) for every
        person above the confidence threshold, largest first.
        """
        h, w = rgb.shape[:2]
        r, new_h, new_w, pad_t, pad_l = self._letterbox_params(h, w)

        # --- preprocess, on the GPU ---
        src = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, non_blocking=True)
        chw = src.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        resized = torch.nn.functional.interpolate(
            chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        self.in_buf.zero_()
        self.in_buf[..., pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized.to(
            self.in_buf.dtype)

        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("YOLO TensorRT execution failed")
        caller.wait_stream(self.stream)

        det = self.outputs[self.det_name][0]          # (N, 38)
        protos = self.outputs[self.proto_name][0]     # (32, ph, pw)

        # Every `.item()`/`bool()`/`int()` on a CUDA tensor is a pipeline flush.
        # Pick the winning row entirely on the GPU and pay for exactly one
        # host transfer at the end; three scattered syncs cost ~4.5 ms here.
        keep = (det[:, 4] >= self.conf) & (det[:, 5].round() == self.person_class)
        areas = (det[:, 2] - det[:, 0]).clamp_min(0) * \
                (det[:, 3] - det[:, 1]).clamp_min(0)
        areas = torch.where(keep, areas, torch.full_like(areas, -1.0))

        if not all_people:
            # Every `.item()`/`bool()`/`int()` on a CUDA tensor is a pipeline
            # flush.  Pick the winning row entirely on the GPU and pay for
            # exactly one host transfer at the end; three scattered syncs cost
            # ~4.5 ms here.
            best = areas.argmax()
            box_src, mask = self._decode(det, protos, best, h, w, r,
                                         pad_t, pad_l, new_h, new_w)
            meta = torch.cat([box_src, areas.index_select(0, best)]).cpu().numpy()
            if meta[4] <= 0:
                return None
            return meta[:4], mask.cpu().numpy()

        # Every candidate, so a caller can show them all and choose.
        idx = torch.nonzero(keep).flatten()
        if not len(idx):
            return []
        order = torch.argsort(areas.index_select(0, idx), descending=True)
        out = []
        for i in idx.index_select(0, order).tolist():
            sel = torch.tensor(i, device=det.device)
            box_src, mask = self._decode(det, protos, sel, h, w, r,
                                         pad_t, pad_l, new_h, new_w)
            conf = float(det[i, 4].item())
            out.append((box_src.cpu().numpy(), mask.cpu().numpy(), conf))
        return out

    def detect_all(self, rgb):
        """Every allowed detection: (box_xyxy, mask HxW bool, conf, class id).

        The segmenter needs furniture as well as people, so unlike __call__
        this keeps the class id and does not reduce to one winner.  Every row
        is decoded in ONE batch: the per-detection loop __call__ uses is fine
        for the one or two people the pose demos look at, but this frame
        carries about fourteen instances, and fourteen separate full-resolution
        interpolations cost more than the ultralytics wrapper this is meant to
        beat.
        """
        h, w = rgb.shape[:2]
        r, new_h, new_w, pad_t, pad_l = self._letterbox_params(h, w)
        self._infer(rgb, new_h, new_w, pad_t, pad_l)

        det = self.outputs[self.det_name][0]          # (N, 38)
        protos = self.outputs[self.proto_name][0]     # (32, ph, pw)

        cls = det[:, 5].round()
        keep = det[:, 4] >= self.conf
        if self.classes is not None:
            allowed = torch.tensor(self.classes, device=det.device,
                                   dtype=cls.dtype)
            keep &= (cls.unsqueeze(1) == allowed).any(1)
        idx = torch.nonzero(keep).flatten()
        if not len(idx):
            return []

        rows = det.index_select(0, idx)               # (K, 38)
        coeffs = rows[:, 6:].float()                  # (K, 32)
        ph, pw = protos.shape[-2:]
        masks = torch.sigmoid(
            coeffs @ protos.float().reshape(protos.shape[0], -1)
        ).reshape(-1, ph, pw)                         # (K, ph, pw)

        sy, sx = ph / self.net_h, pw / self.net_w
        y0, y1 = int(pad_t * sy), int(round((pad_t + new_h) * sy))
        x0, x1 = int(pad_l * sx), int(round((pad_l + new_w) * sx))
        crop = masks[:, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].unsqueeze(1)
        full = torch.nn.functional.interpolate(
            crop, size=(h, w), mode="bilinear", align_corners=False)[:, 0]

        b = rows[:, :4]
        boxes = torch.stack([(b[:, 0] - pad_l) / r, (b[:, 1] - pad_t) / r,
                             (b[:, 2] - pad_l) / r, (b[:, 3] - pad_t) / r], 1)
        boxes[:, 0].clamp_(0, w - 1); boxes[:, 2].clamp_(0, w - 1)
        boxes[:, 1].clamp_(0, h - 1); boxes[:, 3].clamp_(0, h - 1)

        ys = torch.arange(h, device=self.device).view(1, h, 1)
        xs = torch.arange(w, device=self.device).view(1, 1, w)
        bx = boxes.view(-1, 4, 1, 1)
        inside = ((xs >= bx[:, 0]) & (xs <= bx[:, 2]) &
                  (ys >= bx[:, 1]) & (ys <= bx[:, 3]))
        out_masks = (full > 0.5) & inside

        # One transfer for the small tensors, one for the masks.
        meta = torch.cat([boxes, rows[:, 4:5].float(),
                          cls.index_select(0, idx).unsqueeze(1).float()], 1).cpu().numpy()
        mnp = out_masks.cpu().numpy()
        return [(meta[i, :4], mnp[i], float(meta[i, 4]), int(meta[i, 5]))
                for i in range(len(idx))]

    def _infer(self, rgb, new_h, new_w, pad_t, pad_l):
        """Letterbox on the GPU and run the engine."""
        src = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, non_blocking=True)
        chw = src.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        resized = torch.nn.functional.interpolate(
            chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        self.in_buf.zero_()
        self.in_buf[..., pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized.to(
            self.in_buf.dtype)
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("YOLO TensorRT execution failed")
        caller.wait_stream(self.stream)

    def _decode(self, det, protos, row_idx, h, w, r, pad_t, pad_l, new_h, new_w):
        """One detection row -> (box in source pixels, full-res bool mask)."""
        row = det.index_select(0, row_idx.reshape(1))[0]
        box_net, coeffs = row[:4], row[6:].float()

        ph, pw = protos.shape[-2:]
        m = torch.sigmoid((coeffs @ protos.float().reshape(protos.shape[0], -1))
                          .reshape(ph, pw))
        # Prototypes span the letterboxed canvas; crop the real-image window,
        # then resize straight to the source resolution.
        sy, sx = ph / self.net_h, pw / self.net_w
        y0, y1 = int(pad_t * sy), int(round((pad_t + new_h) * sy))
        x0, x1 = int(pad_l * sx), int(round((pad_l + new_w) * sx))
        crop = m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)][None, None]
        full = torch.nn.functional.interpolate(
            crop, size=(h, w), mode="bilinear", align_corners=False)[0, 0]

        box_src = torch.stack([(box_net[0] - pad_l) / r, (box_net[1] - pad_t) / r,
                               (box_net[2] - pad_l) / r, (box_net[3] - pad_t) / r])
        box_src = torch.stack([box_src[0].clamp(0, w - 1), box_src[1].clamp(0, h - 1),
                               box_src[2].clamp(0, w - 1), box_src[3].clamp(0, h - 1)])
        ys = torch.arange(h, device=self.device).unsqueeze(1)
        xs = torch.arange(w, device=self.device).unsqueeze(0)
        inside = ((xs >= box_src[0]) & (xs <= box_src[2]) &
                  (ys >= box_src[1]) & (ys <= box_src[3]))
        return box_src, (full > 0.5) & inside
