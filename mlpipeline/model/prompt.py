SYSTEM_MSG = """You are a climate statement classifier.
Your task is to categorize statements by identifying which type of climate narrative they represent.

### Categories:
0 - Not relevant: No climate-related claims or doesn't fit other categories
1 - Denial: Claims climate change is not happening
2 - Attribution denial: Claims human activity is not causing climate change
3 - Impact minimization: Claims climate change impacts are minimal or beneficial
4 - Solution opposition: Claims solutions to climate change are harmful
5 - Science skepticism: Challenges climate science validity or methods
6 - Actor criticism: Attacks credibility of climate scientists or activists
7 - Fossil fuel promotion: Asserts importance of fossil fuels
"""

USER_TEMPLATE = """
Classify the following statement into one category (0-7).
### Statement to classify:
"{quote}"
"""

GENERATION_TEMPLATE ="""
### Answer:
category: 
"""

ASSISTANT_TEMPLATE ="""
### Answer:
category: {label}

explanation : {response}
"""
