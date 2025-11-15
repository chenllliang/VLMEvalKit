import re
from ...smp import *

Option_list=['A','B','C','D','E','F','G','H','I','J']

llm_judge_prompt="""You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

"""


INPUT_TEMPLATE="""Input:

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""

def extract_last_boxed_content(text):
    """Extract the content from the last \\boxed{} in the text"""
    # Find all \boxed{...} patterns
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last match
        return matches[-1]
    return None



def BabyVision_auxeval(model, line, llm_judge_prompt, INPUT_TEMPLATE):
    """
    Auxiliary evaluation function for single sample evaluation.
    
    Args:
        model: Judge model for evaluation
        line: Data row containing question, answer, prediction fields
        llm_judge_prompt: System prompt for LLM judge
        INPUT_TEMPLATE: Input template for formatting
    
    Returns:
        dict: Contains hit (correctness), extracted_answer, and judge_response
    """
    try:
        prediction = str(line['prediction'])
        extracted_answer = extract_last_boxed_content(prediction)
        
        if extracted_answer is None:
            extracted_answer = "NA"
        
        question = str(line['question']) if 'question' in line else str(line.get('query', ''))
        
        groundtruth = line['answer']

        
        user_input = INPUT_TEMPLATE.format(
            question=question,
            groundtruth=groundtruth,
            modeloutput=extracted_answer
        )
        
        full_prompt = llm_judge_prompt + "\n\n" + user_input
        
        judge_response = model.generate(full_prompt, temperature=0)
        
        judge_response_clean = str(judge_response).strip().lower()
        
        if 'true' in judge_response_clean:
            hit = 1
        elif 'false' in judge_response_clean:
            hit = 0
        else:
            hit = 0
            print(f"Warning: Unable to parse judge response: {judge_response}")
        
        return {
            'hit': hit,
            'extracted_answer': extracted_answer,
            'judge_response': judge_response
        }
        
    except Exception as e:
        print(f"Error in BabyVision_auxeval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'hit': 0,
            'extracted_answer': '',
            'judge_response': f'Error: {str(e)}'
        }


def BabyVisionACC(result_file):
    """
    Calculate accuracy for BabyVision evaluation results.
    
    Args:
        result_file: Path to the file containing evaluation results
    
    Returns:
        dict: Dictionary containing overall and per-category accuracy scores
    """
    data = load(result_file)
    
    result_dict = {}
    
    if 'hit' in data.columns:
        result_dict['Overall'] = data['hit'].mean() * 100
    
    if 'category' in data.columns:
        for cat in data['category'].unique():
            if pd.notna(cat) and str(cat).strip():
                cat_data = data[data['category'] == cat]
                result_dict[str(cat)] = cat_data['hit'].mean() * 100
    
    return result_dict