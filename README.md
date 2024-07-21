# CRAG Application for Game Lore analysis.

This code creates a CRAG application following the [paper](https://arxiv.org/abs/2401.15884). The only difference is the models used and the refinement process. In this case, retrieved information is compressed using LLMChainExtractor from LangChain.

### Instruments used:

- LangGraph for building a CRAG application
- LangChain for adding additional troubleshooting mechanisms
- Tavily for web searching
- Chroma database for vector database
- Chainlit for interface
- LangSmith for logging

### Model used:

- gpt-4o-mini

## Troubleshooting

To verify that model outputs could be parsed without errors RetryWithErrorOutputParser was utilised. During document grading, the model should provide a number for each document, a special parser was created to ensure that. If the model fails to follow the parser's format instructions, an exception will occur and this problem will be handled by the RetryWithErrorOutputParser which will update the model's output following the initial instructions.
Apart from the response format, the content is verified with the help of ConstitutionalPrinciple. Two principles that the model should follow were established:

1. Ethical Principle: The model should be polite and never insult the user
2. Game Image Principle: The model should never say anything bad about the Genshin Impact game. It can say bad things about characters, but it should not critique the game itself

These two instruments helped to make the model's output more robust both in terms of its format and content.
