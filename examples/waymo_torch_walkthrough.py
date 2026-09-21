import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.patches import Polygon, Rectangle
    from PIL import Image
    from torch.utils.data import DataLoader

    from torch_waymo import WaymoDataset
    from torch_waymo.protocol.dataset_proto import CameraName

    return (
        CameraName,
        DataLoader,
        Image,
        Path,
        Polygon,
        Rectangle,
        WaymoDataset,
        io,
        mo,
        np,
        os,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🚗 Loading & Visualizing Waymo Open Dataset with `torch_waymo`

    This reactive **Marimo notebook** demonstrates how to load, inspect, visualize, and convert
    [Waymo Open Dataset](https://waymo.com/open/) Perception frames into **PyTorch tensors**.

    ### Key Features Demonstrated:
    1. **Dataset Loading**: Load converted Waymo v2 Parquet (`.pkl.gz`) or v1 TFRecord (`.pkl`) frames with `WaymoDataset`.
    2. **Multi-Camera Rendering**: Decode JPEG image bytes into PyTorch `(3, H, W)` tensors & overlay 2D/3D labels.
    3. **LiDAR Point Cloud Visualization**: Extract 3D points, color by height/range, and draw oriented 3D bounding boxes in Bird's-Eye View (BEV).
    4. **Range Image & Segmentation**: Inspect range images (depth maps) and panoptic camera segmentation labels.
    5. **PyTorch `DataLoader` Integration**: Collate frames into PyTorch model input batches using custom transforms.
    """)
    return


@app.cell
def _(mo, os):
    dataset_path_input = mo.ui.text(
        value=os.environ.get(
            "TORCH_WAYMO_DATASET",
            "~/Datasets/Waymo-v2-subset/converted",
        ),
        label="Converted dataset root directory",
        full_width=True,
    )
    split_select = mo.ui.dropdown(
        options=["training", "validation", "testing"],
        value="training",
        label="Dataset Split",
    )

    mo.hstack([dataset_path_input, split_select], justify="start", align="center")
    return dataset_path_input, split_select


@app.cell
def _(Path, WaymoDataset, dataset_path_input, mo, split_select):
    dataset_root = Path(dataset_path_input.value).expanduser()
    split_name = split_select.value
    split_path = dataset_root / split_name

    dataset = None
    load_error_msg = None

    if not dataset_root.exists() or not dataset_root.is_dir():
        load_error_msg = f"Dataset root path does not exist: `{dataset_root}`"
    elif not split_path.exists() or not split_path.is_dir():
        load_error_msg = f"Split directory `{split_name}` does not exist in `{dataset_root}`."
    elif not (split_path / "len.pkl").exists():
        load_error_msg = f"No index cache (`len.pkl`) found in `{split_path}`. The dataset may not be converted yet."
    else:
        try:
            dataset = WaymoDataset(dataset_root, split_name)
        except Exception as err:
            load_error_msg = str(err)

    if dataset is not None:
        frame_index_slider = mo.ui.slider(
            start=0,
            stop=len(dataset) - 1,
            step=1,
            value=0,
            label="Select sample frame index",
            show_value=True,
        )
        status_view = mo.vstack(
            [
                mo.md(f"✅ Loaded **{len(dataset)} frames** from `{dataset_root}` (`{split_name}` split)."),
                frame_index_slider,
            ]
        )
    else:
        frame_index_slider = None
        status_view = mo.callout(
            mo.md(
                f"""
                ### ⚠️ Converted Dataset Not Found
                **{load_error_msg}**

                To download and convert the one-segment v2 subset (see `README.md`):

                ```shell
                # Convert dataset:
                uv sync --extra waymo
                uv run torch-waymo-convert --dataset ~/Datasets/Waymo-v2-subset --splits training
                ```
                """
            ),
            kind="warn",
        )

    status_view
    return dataset, dataset_root, frame_index_slider


@app.cell
def _(dataset, frame_index_slider, mo):
    if dataset is None or frame_index_slider is None:
        sample_info_output = mo.md("⚠️ *Provide a valid converted dataset path above to view sample frames.*")
        frame = None
        cam_count = 0
    else:
        frame = dataset[frame_index_slider.value]

        cam_count = len(frame.images) if hasattr(frame, "images") and frame.images is not None else 0
        laser_count = len(getattr(frame, "lasers", []))
        label_count = len(getattr(frame, "laser_labels", []))
        total_pts = sum(len(p) for p in frame.points) if hasattr(frame, "points") and frame.points is not None else 0
        segment_name = getattr(getattr(frame, "context", None), "name", "N/A")
        timestamp = getattr(frame, "timestamp_micros", "N/A")

        sample_info_output = mo.md(
            f"""
            ### 📊 Sample Frame `{frame_index_slider.value}` Summary
            - **Segment Name:** `{segment_name}`
            - **Timestamp (µs):** `{timestamp}`
            - **Sensors:** `{cam_count}` cameras | `{laser_count}` LiDAR sensors
            - **Annotations:** `{label_count}` 3D objects
            - **Point Cloud:** `{total_pts:,}` 3D points
            """
        )

    sample_info_output
    return cam_count, frame


@app.cell
def _(cam_count, dataset, mo):
    if dataset is None or cam_count == 0:
        camera_controls_output = mo.md("")
        camera_select = None
        camera_view_mode = None
    else:
        camera_view_mode = mo.ui.dropdown(
            options=["Single Camera View", "All 5 Cameras Grid View"],
            value="Single Camera View",
            label="Camera Display Mode",
        )
        camera_select = mo.ui.slider(
            start=0,
            stop=max(0, cam_count - 1),
            step=1,
            value=0,
            label="Camera Index",
            show_value=True,
        )

        camera_controls_output = mo.hstack([camera_view_mode, camera_select], justify="start", align="center")

    camera_controls_output
    return camera_select, camera_view_mode


@app.cell
def _(
    CameraName,
    Image,
    Rectangle,
    camera_select,
    camera_view_mode,
    frame,
    io,
    mo,
    np,
    plt,
    torch,
):
    if frame is None or not hasattr(frame, "images") or frame.images is None or len(frame.images) == 0:
        camera_output = mo.md("⚠️ *No camera images available in this frame (e.g. SimplifiedFrame).*")
    else:
        images_list = frame.images
        if camera_view_mode.value == "Single Camera View":
            idx = min(camera_select.value, len(images_list) - 1)
            cam_item = images_list[idx]

            try:
                cam_enum = CameraName(cam_item.name).name
            except Exception:
                cam_enum = str(cam_item.name)

            full_img = Image.open(io.BytesIO(cam_item.image)).convert("RGB")
            orig_w, orig_h = full_img.size
            cam_arr = np.array(full_img)
            cam_tensor = torch.from_numpy(cam_arr).permute(2, 0, 1)

            # Downsample for display to keep marimo payload compact
            max_disp = 960
            if orig_w > max_disp or orig_h > max_disp:
                scale = max_disp / max(orig_w, orig_h)
                disp_img = full_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.BILINEAR)
            else:
                scale = 1.0
                disp_img = full_img

            cam_labels = next(
                (g.labels for g in getattr(frame, "camera_labels", []) if int(g.name) == int(cam_item.name)),
                [],
            )

            fig, ax = plt.subplots(figsize=(10, 6), dpi=90)
            ax.imshow(np.array(disp_img))
            for lbl in cam_labels:
                _box = lbl.box
                _cx, _cy = _box.center_x * scale, _box.center_y * scale
                _bl, _bw = _box.length * scale, _box.width * scale
                rect = Rectangle(
                    (_cx - _bl / 2, _cy - _bw / 2),
                    _bl,
                    _bw,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
                ax.add_patch(rect)

            ax.set_title(f"Camera {cam_enum} (Index {cam_item.name}) | {len(cam_labels)} 2D boxes", fontsize=11)
            ax.axis("off")
            plt.tight_layout()

            camera_output = mo.vstack(
                [
                    mo.md(
                        f"## 📷 Camera View: `{cam_enum}`\n"
                        f"**Tensor Shape:** `{tuple(cam_tensor.shape)}` | **Dtype:** `{cam_tensor.dtype}` | **Resolution:** `{orig_w}x{orig_h}`"
                    ),
                    fig,
                ]
            )
        else:
            # 5 Cameras Grid View
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=90)
            axes = axes.flatten()

            for i, cam_item in enumerate(images_list):
                try:
                    cname = CameraName(cam_item.name).name
                except Exception:
                    cname = str(cam_item.name)
                full_img = Image.open(io.BytesIO(cam_item.image)).convert("RGB")
                orig_w, orig_h = full_img.size
                max_grid = 480
                scale = max_grid / max(orig_w, orig_h)
                grid_img = full_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.BILINEAR)
                axes[i].imshow(np.array(grid_img))
                axes[i].set_title(cname, fontsize=10)
                axes[i].axis("off")

            # Turn off unused 6th subplot
            axes[5].axis("off")
            plt.suptitle(f"All {len(images_list)} Camera Images Grid", fontsize=13)
            plt.tight_layout()

            camera_output = mo.vstack(
                [
                    mo.md("## 📷 Multi-Camera Array (5 Views)"),
                    fig,
                ]
            )

    camera_output
    return


@app.cell
def _(Polygon, frame, mo, np, plt, torch):
    if frame is None:
        bev_output = mo.md("⚠️ *No sample frame loaded.*")
    else:
        # Extract LiDAR points and 3D bounding boxes
        raw_points_list = getattr(frame, "points", None) or []
        pts_tensors = [torch.from_numpy(np.asarray(p)).to(torch.float32) for p in raw_points_list if len(p) > 0]

        if pts_tensors:
            points_all = torch.cat(pts_tensors, dim=0)
        else:
            points_all = torch.zeros((0, 3), dtype=torch.float32)

        raw_labels = getattr(frame, "laser_labels", []) or []
        boxes_3d = torch.tensor(
            [
                [
                    lbl.box.center_x,
                    lbl.box.center_y,
                    lbl.box.center_z,
                    lbl.box.length,
                    lbl.box.width,
                    lbl.box.height,
                    lbl.box.heading,
                ]
                for lbl in raw_labels
            ],
            dtype=torch.float32,
        )
        box_types = torch.tensor([int(lbl.type) for lbl in raw_labels], dtype=torch.int64)

        # BEV Visualization
        if len(points_all) > 0:
            stride = max(1, len(points_all) // 25_000)
            pts_sampled = points_all[::stride].numpy()

            fig_bev, ax_bev = plt.subplots(figsize=(8, 8), dpi=90)
            sc = ax_bev.scatter(
                pts_sampled[:, 0],
                pts_sampled[:, 1],
                s=0.3,
                c=pts_sampled[:, 2],
                cmap="viridis",
            )
            plt.colorbar(sc, ax=ax_bev, label="Height Z (meters)")

            # Draw oriented 3D boxes in BEV
            type_colors = {1: "cyan", 2: "red", 3: "yellow", 4: "orange"}
            for b, t in zip(boxes_3d.numpy(), box_types.numpy(), strict=False):
                _cx, _cy, _, length, width, _, heading = b
                c, s = np.cos(heading), np.sin(heading)
                rot = np.array([[c, -s], [s, c]])
                local_corners = np.array(
                    [
                        [length / 2, width / 2],
                        [length / 2, -width / 2],
                        [-length / 2, -width / 2],
                        [-length / 2, width / 2],
                    ]
                )
                world_corners = local_corners @ rot.T + np.array([_cx, _cy])
                color = type_colors.get(int(t), "magenta")

                polygon = Polygon(
                    world_corners,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                )
                ax_bev.add_patch(polygon)

                # Heading direction indicator line
                front_mid = np.array([length / 2, 0.0]) @ rot.T + np.array([_cx, _cy])
                ax_bev.plot([_cx, front_mid[0]], [_cy, front_mid[1]], color=color, linewidth=2)

            ax_bev.scatter([0], [0], color="red", marker="x", s=80, label="Ego Vehicle Origin")
            ax_bev.set_xlabel("Vehicle X (Forward, meters)")
            ax_bev.set_ylabel("Vehicle Y (Left, meters)")
            ax_bev.set_title(
                f"LiDAR Bird's-Eye View (BEV) | {len(points_all):,} points, {len(boxes_3d)} 3D boxes", fontsize=11
            )
            ax_bev.set_aspect("equal", adjustable="datalim")
            ax_bev.grid(True, linestyle="--", alpha=0.3)
            ax_bev.legend(loc="upper right")
            plt.tight_layout()

            bev_output = mo.vstack(
                [
                    mo.md(
                        f"## 🧊 LiDAR Point Cloud & 3D Bounding Boxes\n"
                        f"- **Points Tensor:** `{tuple(points_all.shape)}` `{points_all.dtype}`\n"
                        f"- **3D Boxes Tensor `[x, y, z, l, w, h, heading]`:** `{tuple(boxes_3d.shape)}`\n"
                        f"- **Box Classes Tensor:** `{tuple(box_types.shape)}`"
                    ),
                    fig_bev,
                ]
            )
        else:
            bev_output = mo.md("⚠️ *No LiDAR point cloud data available for this frame.*")

    bev_output
    return


@app.cell
def _(frame, mo, np, plt, torch):
    if frame is None:
        ri_output = mo.md("")
    else:
        lasers = getattr(frame, "lasers", []) or []
        top_laser = next(
            (laser_item for laser_item in lasers if int(laser_item.name) == 1), lasers[0] if lasers else None
        )

        if top_laser is not None and getattr(top_laser, "ri_return1", None) is not None:
            first_ret = top_laser.ri_return1
            if first_ret.values is not None:
                ri_val = torch.from_numpy(np.array(first_ret.values, copy=True)).to(torch.float32)
                valid_px = int((ri_val[..., 0] > 0).sum())

                fig_ri, ax_ri = plt.subplots(figsize=(11, 3.5), dpi=90)
                range_channel = ri_val[..., 0].numpy()
                im = ax_ri.imshow(range_channel, aspect="auto", cmap="magma")
                plt.colorbar(im, ax=ax_ri, label="Range (m)")
                ax_ri.set_title(f"Top LiDAR Range Image (Return 1) | Shape {tuple(ri_val.shape)}")
                ax_ri.set_xlabel("Azimuth Col")
                ax_ri.set_ylabel("Inclination Row")
                plt.tight_layout()

                ri_output = mo.vstack(
                    [
                        mo.md(
                            f"## 📡 LiDAR Range Image (Depth Map)\n"
                            f"- **Range Image Shape:** `{tuple(ri_val.shape)}` | Valid returns: **{valid_px:,}**"
                        ),
                        fig_ri,
                    ]
                )
            else:
                ri_output = mo.md(
                    "## 📡 LiDAR Range Image\n"
                    f"Compressed payload present (**{len(first_ret.range_image_compressed or b''):,} bytes**)."
                )
        else:
            ri_output = mo.md("## 📡 LiDAR Range Image\nNo range image data available in frame.")

    ri_output
    return


@app.cell
def _(Image, dataset, io, mo, np, plt, torch):
    if dataset is None:
        seg_output = mo.md("")
    else:
        segmentation_example = None
        for _frame_index in range(min(len(dataset), 50)):
            for _candidate in dataset[_frame_index].images:
                _panoptic = getattr(_candidate.camera_segmentation_label, "panoptic_label", None)
                if _panoptic is not None and len(_panoptic) > 0:
                    segmentation_example = (_frame_index, _candidate, _panoptic)
                    break
            if segmentation_example is not None:
                break

        if segmentation_example is not None:
            seg_frame_idx, seg_cam, panoptic_bytes = segmentation_example
            if isinstance(panoptic_bytes, (bytes, bytearray)):
                panoptic_arr = np.array(Image.open(io.BytesIO(panoptic_bytes)), copy=True)
            else:
                panoptic_arr = np.asarray(panoptic_bytes)
            panoptic_tensor = torch.from_numpy(panoptic_arr.astype(np.int64, copy=False))

            fig_seg, ax_seg = plt.subplots(figsize=(10, 5), dpi=90)
            ax_seg.imshow(panoptic_tensor.numpy())
            ax_seg.set_title(f"Panoptic Segmentation: Frame {seg_frame_idx}, Camera {seg_cam.name}")
            ax_seg.axis("off")
            plt.tight_layout()

            seg_output = mo.vstack(
                [
                    mo.md(
                        f"## 🧩 Camera Panoptic Segmentation\n"
                        f"**Tensor Shape:** `{tuple(panoptic_tensor.shape)}` | **Dtype:** `{panoptic_tensor.dtype}`"
                    ),
                    fig_seg,
                ]
            )
        else:
            seg_output = mo.md(
                "## 🧩 Camera Panoptic Segmentation\n*No segmentation labels found in the first 50 frames (segmentation labels are sparse).* "
            )

    seg_output
    return


@app.cell
def _(
    DataLoader,
    Image,
    WaymoDataset,
    dataset,
    dataset_root,
    io,
    mo,
    np,
    split_select,
    torch,
):
    if dataset is None:
        loader_output = mo.md("")
    else:

        def frame_to_pytorch_dict(item):
            """Converts a Waymo frame object into a clean dictionary of PyTorch Tensors."""
            out = {
                "timestamp_micros": item.timestamp_micros,
            }

            # Camera tensor (C, H, W)
            if hasattr(item, "images") and len(item.images) > 0:
                cam_arr = np.array(Image.open(io.BytesIO(item.images[0].image)).convert("RGB"))
                out["front_camera"] = torch.from_numpy(cam_arr).permute(2, 0, 1)

            # Concatenated LiDAR point cloud (N, 3)
            if hasattr(item, "points") and item.points is not None:
                pts = [torch.from_numpy(np.asarray(p)).to(torch.float32) for p in item.points if len(p) > 0]
                out["points"] = torch.cat(pts, dim=0) if pts else torch.zeros((0, 3), dtype=torch.float32)

            # 3D Bounding boxes (M, 7) and class targets (M,)
            if hasattr(item, "laser_labels") and item.laser_labels:
                out["boxes_3d"] = torch.tensor(
                    [
                        [
                            lbl.box.center_x,
                            lbl.box.center_y,
                            lbl.box.center_z,
                            lbl.box.length,
                            lbl.box.width,
                            lbl.box.height,
                            lbl.box.heading,
                        ]
                        for lbl in item.laser_labels
                    ],
                    dtype=torch.float32,
                )
                out["labels_3d"] = torch.tensor([int(lbl.type) for lbl in item.laser_labels], dtype=torch.int64)

            return out

        transformed_dataset = WaymoDataset(dataset_root, split_select.value, transform=frame_to_pytorch_dict)
        loader = DataLoader(transformed_dataset, batch_size=None, shuffle=False)
        sample_dict = next(iter(loader))
        summary_repr = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v for k, v in sample_dict.items()}

        loader_output = mo.md(
            f"""
            ## ⚡ PyTorch `DataLoader` Integration

            Passing a `transform` function to `WaymoDataset` allows direct integration with standard
            PyTorch pipelines.

            ```python
            from torch.utils.data import DataLoader
            from torch_waymo import WaymoDataset

            dataset = WaymoDataset(dataset_root, split="training", transform=frame_to_pytorch_dict)
            loader = DataLoader(dataset, batch_size=None, shuffle=True)
            ```

            **Sample Batch Output from `DataLoader`:**
            `{summary_repr}`
            """
        )

    loader_output
    return


if __name__ == "__main__":
    app.run()
