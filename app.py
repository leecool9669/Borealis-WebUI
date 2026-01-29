import gradio as gr
from PIL import Image, ImageFilter
from typing import Tuple, Optional


def dummy_infer(
    image: Optional[Image.Image],
    prompt: str,
    strength: float,
    seed: int,
) -> Tuple[Optional[Image.Image], str]:
    """
    该函数并不真正调用 Borealis 模型，而是对输入图像做轻量级滤波处理，
    用于演示 WebUI 的交互流程与可视化效果，避免在本地下载大规模权重。
    """
    if image is None:
        return None, "请先上传一张测试图像。"

    # 伪造的“增强”效果：边缘增强 + 轻微平滑，仅用于可视化演示
    enhanced = image.convert("RGB").filter(ImageFilter.DETAIL).filter(ImageFilter.SMOOTH_MORE)

    description = (
        "伪推理完成：当前为演示模式，并未真正加载 Borealis 权重。\n"
        f"提示词：{prompt or '（未填写）'}；"
        f"结构保持系数：{strength:.2f}；"
        f"随机种子：{seed}。\n"
        "在真实部署环境中，可在此处接入 Borealis 推理接口，以获得物理一致的图像增强结果。"
    )
    return enhanced, description


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Borealis 图像增强 WebUI 演示") as demo:
        gr.Markdown(
            """
            # Borealis 图像增强 WebUI（演示版）

            本界面用于演示 Borealis 模型在图像增强任务中的典型交互流程，
            包括图像上传、参数配置、推理触发与结果可视化等关键步骤。
            出于资源考虑，本示例未在前端实际下载和加载完整模型，仅进行轻量级图像处理以模拟推理效果。
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="输入图像", type="pil")
                prompt = gr.Textbox(
                    label="物理先验 / 文本提示（可选）",
                    placeholder="例如：保持星空结构，抑制噪声，增强局部对比度……",
                )
                strength = gr.Slider(
                    label="结构保持系数",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                )
                seed = gr.Slider(
                    label="随机种子",
                    minimum=0,
                    maximum=2**31 - 1,
                    value=42,
                    step=1,
                )
                run_button = gr.Button("运行推理（演示）")

            with gr.Column():
                output_image = gr.Image(label="输出图像（伪增强结果）")
                info = gr.Textbox(
                    label="推理说明",
                    interactive=False,
                    lines=6,
                )

        run_button.click(
            fn=dummy_infer,
            inputs=[input_image, prompt, strength, seed],
            outputs=[output_image, info],
        )

        gr.Markdown(
            """
            **说明**：实际接入 Borealis 模型时，只需在后端将 `dummy_infer` 替换为真正的推理函数，
            并在其中完成模型加载、数据预处理与物理一致性的后处理即可。
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860)
