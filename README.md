# CultureWeave - GenAI for Indigenous Knowledge Preservation

A Jac application that demonstrates scale-agnostic design for indigenous knowledge preservation and storytelling.

## Features

### StoryWeaver Walker
- **Story Generation**: Creates personalized indigenous stories based on user prompts
- **Multi-language Support**: Supports English, Kiswahili, Kikuyu, Luo, Kamba, and Maasai
- **Cultural Themes**: Incorporates traditional themes like wisdom, courage, family, nature, spirits, and ancestors
- **Translation**: Automatically translates stories to native languages

### KnowledgeKeeper Walker
- **Cultural Preservation**: Stores and manages cultural knowledge elements
- **Elder Verification**: Simulates community elder verification process
- **Quality Rating**: Provides community-based quality ratings for cultural content
- **Verification Queue**: Manages content that needs further review

## Scale-Agnostic Design

This application demonstrates Jac's scale-agnostic approach:

### Local CLI Mode
```bash
# Run locally for testing
jac run cultureweave.jac
```

### Cloud Deployment
```bash
# Deploy as API endpoints
jac serve cultureweave.jac
```

The same code runs both locally and in the cloud without modification. When served, the walkers become HTTP API endpoints that can be called via REST requests.

## Usage Examples

### Story Generation
The application generates stories like:
- "In the ancient times, when the spirits walked among us... Tell me a story about wisdom from our ancestors. The story teaches us about courage and connects us to our heritage."

### Knowledge Preservation
- Preserves cultural elements like "Traditional farming techniques"
- Verifies content through simulated elder review process
- Maintains quality ratings and verification status

## Architecture

- **Nodes**: `story_node`, `translation_node`, `knowledge_node`, `verification_node`
- **Walkers**: `StoryWeaver`, `KnowledgeKeeper`
- **Entry Point**: CLI mode with sample data initialization

## Cultural Impact

This minimal implementation demonstrates how technology can:
- Preserve indigenous knowledge and traditions
- Support multi-language storytelling
- Enable community-driven content verification
- Scale from local communities to global platforms

The application serves as a foundation for building more comprehensive cultural preservation systems that can help address the challenges of cultural erosion and language extinction.
