# SafePrompt - ComfyUI的AI提示词清洗节点

这是一个为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 设计的自定义节点，它利用智谱AI（Zhipu AI）的强大语言模型，根据您提供的自然语言指令，智能地清洗、优化或重写您的提示词（Prompt）。

---

## ✨ 功能特性

- **AI驱动的文本处理**: 不仅仅是简单的文本替换，它能理解您的意图并进行智能修改。
- **灵活的指令**: 您可以通过简单的指令（如“删除所有NSFW内容”、“将风格改为赛博朋克”、“翻译成英文”）来控制处理过程。
- **模型可选**: 支持多种智谱AI模型，包括 `glm-4.5-flash`, `glm-4.5`, 和 `glm-z1-flash`等。
- **安全配置**: API密钥通过独立的配置文件管理，不会暴露在工作流或节点界面上。

---

## 🚀 安装指南

1. **获取节点文件**

   - 通过 `git clone` 或直接下载ZIP包的方式，将本仓库下载到本地。
2. **放置节点**

   - 将 `SafePrompt` 文件夹完整地放入 ComfyUI 的自定义节点目录中：
     ```bash
     ComfyUI/custom_nodes/
     ```
3. **安装依赖**

   - 打开您的终端（命令行工具），进入 `SafePrompt` 目录：
     ```bash
     cd ComfyUI/custom_nodes/SafePrompt
     ```
   - 安装所需的Python库：
     ```bash
     pip install -r requirements.txt
     ```
4. **配置API密钥**

   - 在 `SafePrompt` 文件夹中，找到 `config.json.example` 文件。
   - **复制**该文件并将其**重命名**为 `config.json`。
   - 用文本编辑器打开 `config.json`，将 `"YOUR_ZHIPU_API_KEY_HERE"` 替换为您自己的智谱AI API密钥。
5. **重启ComfyUI**

   - 关闭并重新启动ComfyUI。您现在应该可以在节点菜单中找到它。

---

## 📝 如何使用

1. **添加节点**: 在ComfyUI的节点菜单中，选择 `SafePrompt` -> `SafePrompt` 即可将节点添加到您的工作流中。
2. **连接输入**:

   - `prompt`: 连接您想要处理的原始提示词。这可以来自一个“Primitive”节点或任何其他输出文本的节点。
   - `instruction`: 在此文本框中输入您希望AI执行的指令。例如：“请把所有关于猫的描述都换成狗”。
   - `model`: 从下拉菜单中选择一个AI模型。
   - `seed`: 随机种子，用于确保AI在同样输入下能产生可复现的结果。
3. **获取输出**:

   - `cleaned_prompt`: 此输出端口将提供经过AI处理后的、干净的提示词。您可以将其连接到任何需要提示词输入的节点，例如KSampler。

### 示例工作流

```
[Primitive String] -> [SafePrompt] -> [CLIP Text Encode]
      (prompt)    ->  (prompt)      (cleaned_prompt)

[Primitive String] ->  (instruction)
   (instruction)
```

---

祝您使用愉快！
