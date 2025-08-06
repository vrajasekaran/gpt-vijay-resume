---
title: Vijay Resume GPT
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
- resume
- gpt
- ai
- chatbot
- transformer
- pytorch
---

# 🤖 Chat with Vijay Rajasekaran's Resume GPT

This is a **custom-trained mini-GPT model** that can answer questions about Vijay Rajasekaran's professional background. The model was built from scratch using a transformer architecture and trained exclusively on resume data.

## 🚀 Features

- **Custom Transformer Architecture**: Built from scratch with PyTorch
- **Specialized Knowledge**: Trained on 75+ Q&A pairs about Vijay's experience
- **Interactive Chat Interface**: Real-time conversations about professional background
- **Lightweight Model**: Only ~2M parameters for fast inference
- **Fallback Responses**: Demo mode with pre-programmed answers

## 🎯 What You Can Ask

### Professional Experience
- Current role and responsibilities at EquiB
- Previous positions at BlockApps, Deloitte, and other companies
- Specific projects and achievements

### Technical Skills
- AI/ML frameworks (AutoGen, LangChain, CrewAI)
- Programming languages (Python, JavaScript, C#, etc.)
- Cloud and DevOps technologies
- Database and infrastructure experience

### AI Expertise
- RAG (Retrieval-Augmented Generation) systems
- Multi-agent AI architectures
- LLM fine-tuning and evaluation
- Vector databases and embeddings

### Contact & Background
- Educational background
- Certifications and continuous learning
- Contact information and social profiles

## 🛠 Technical Details

### Model Architecture
- **Type**: Custom Mini-GPT Transformer
- **Parameters**: ~2 million
- **Layers**: 6 transformer blocks
- **Attention Heads**: 8
- **Embedding Dimension**: 256
- **Context Length**: 512 tokens

### Training Data
- **Source**: Vijay Rajasekaran's professional resume
- **Format**: Question-Answer pairs with conversation formatting
- **Size**: 75+ training examples after augmentation
- **Tokenizer**: Custom vocabulary built from resume content

### Performance
- **Response Time**: < 2 seconds on CPU
- **Model Size**: ~8MB on disk
- **Accuracy**: Specialized knowledge about Vijay's background
- **Fallback**: Demo responses when model unavailable

## 🎮 Example Interactions

```
Q: What's your current role?
A: I'm currently a Tech Lead at EquiB in Atlanta, GA, working on an AI AgenticRAG platform for medical equipment financing.

Q: Tell me about your AI experience
A: I specialize in developing autonomous agents, RAG pipelines, and tool-augmented LLM workflows using frameworks like AutoGen, CrewAI, and LangChain...

Q: What did you build at EquiB?
A: I built a sophisticated multi-agent orchestration system with specialized agents for persona analysis, lender matching, and explanation generation...
```

## 🔧 Development Process

This model demonstrates how to:
1. **Extract training data** from resume content
2. **Build custom tokenizers** for domain-specific vocabulary
3. **Implement transformer architecture** from scratch
4. **Train specialized models** on small datasets
5. **Deploy to Hugging Face Spaces** with Gradio

## 📊 Model Comparison

| Aspect | This Model | General LLMs |
|--------|------------|--------------|
| Size | 2M params | 7B+ params |
| Training Data | Resume-specific | General web data |
| Response Quality | High (domain) | Variable |
| Inference Speed | Very Fast | Slower |
| Deployment Cost | Very Low | Higher |

## 🚀 Use Cases

- **Resume Enhancement**: Interactive resume experiences
- **Recruitment**: 24/7 candidate information access
- **Networking**: Engaging professional introductions
- **Portfolio Websites**: Dynamic "About Me" sections
- **Career Services**: Template for student resume bots

## 📝 About Vijay Rajasekaran

**Vijay Rajasekaran** is a Solution Architect specializing in AI and Cloud technologies, currently serving as Tech Lead at EquiB. With extensive experience in:

- **AI Systems**: RAG pipelines, multi-agent architectures, LLM workflows
- **Cloud Architecture**: Azure, microservices, event-driven systems
- **Leadership**: Cross-functional teams, agile delivery, offshore management
- **Innovation**: Emerging AI tools, next-gen agent frameworks

**Contact**: vprajasekaran@gmail.com | +1 (860) 652-5581
**LinkedIn**: [vijayparamasivam](https://www.linkedin.com/in/vijayparamasivam/)
**GitHub**: [vrajasekaran](https://github.com/vrajasekaran)

---

*This space showcases how to build personalized AI assistants using custom transformer models. The approach can be adapted for any professional or personal use case.*