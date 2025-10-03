# CultureWeave - GenAI for Indigenous Knowledge Preservation

A Jac application that demonstrates scale-agnostic design for indigenous knowledge preservation and storytelling with AI-powered features.

## Features

### StoryWeaver Walker
- **AI-Powered Story Generation**: Creates personalized indigenous stories using LLM
- **Multi-language Support**: Built-in menu (English, Kiswahili, Kikuyu, Luo, Kamba, Maasai) + **Other** to type any language
- **Cultural Themes**: Incorporates traditional themes like wisdom, courage, family, nature, spirits, and ancestors
- **AI Translation**: Context-aware translation with cultural nuances

### KnowledgeKeeper Walker
- **Cultural Preservation**: Stores and manages cultural knowledge elements
- **AI Verification**: Intelligent cultural authenticity verification
- **Elder Verification**: Community elder verification process
- **Digital Archiving**: Process and archive cultural artifacts from images
- **Sacred Content Protection**: Special handling for sacred cultural content

## Scale-Agnostic Design

This application demonstrates Jac's scale-agnostic approach:

### Local CLI Mode
```bash
# Activate virtual environment and run
source jac-env/bin/activate  # if created
jac run main.jac
```

### Cloud Deployment
```bash
# Serve as HTTP endpoints
jac serve main.jac
```

The same code runs both locally and in the cloud without modification. When served, the walkers become HTTP API endpoints that can be called via REST requests.

## Setup

1) Create a virtual environment (recommended):
```bash
python3 -m venv jac-env
source jac-env/bin/activate
pip install -U jaclang litellm groq tiktoken requests
```

2) Configure environment variables:
```bash
export GROQ_API_KEY=your_groq_key
export SERPER_API_KEY=your_serper_key
```
You will be prompted for keys at runtime if they are not set.

3) Run:
```bash
jac run main.jac
```

## Usage Examples

### Story Generation
The application generates stories like:
- "In the ancient times, when the spirits walked among us... Tell me a story about wisdom from our ancestors. The story teaches us about courage and connects us to our heritage."

### Knowledge Preservation
- Preserves cultural elements like "Traditional farming techniques"
- Verifies content through simulated elder review process
- Maintains quality ratings and verification status

## Architecture

### Modular File Structure
- `ai_functions.jac` — AI/LLM integrations
- `nodes.jac` — Node definitions and storage
- `walkers.jac` — Walker class definitions
- `implementations.jac` — Ability implementations for walkers and nodes
- `config.jac` — API key setup/validation
- `main.jac` — Interactive CLI entrypoint

### Core Graph Elements
- **Nodes**: `story_node`, `translation_node`, `knowledge_node`, `verification_node`
- **Walkers**: `StoryWeaver`, `KnowledgeKeeper`

## Cultural Impact

This minimal implementation demonstrates how technology can:
- Preserve indigenous knowledge and traditions
- Support multi-language storytelling
- Enable community-driven content verification
- Scale from local communities to global platforms

The application serves as a foundation for building more comprehensive cultural preservation systems that can help address the challenges of cultural erosion and language extinction.

## Notes

- `jac-env/` (virtual environment) is ignored and should not be committed.
- Language selection menus include an **Other** option; you can type any target language.
