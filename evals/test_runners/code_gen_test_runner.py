from typing import Dict, List, Optional
from evals.ai_api_calls.get_open_ai_completion import get_open_ai_completion
from evals.eval_types import ChatPromptGenerator, CodeGenTestCase, TestCaseResult
from evals.prompts.chat_prompts import CHAT_PROMPT_GENERATORS
from evals.test_cases.code_gen_tests import CODE_GEN_TESTS
from evals.utils import are_globals_equal, get_globals_to_compare, get_script_from_cells, print_test_case_result_tables


def run_code_gen_tests(test_name: Optional[str], prompt_name: Optional[str], tags: Optional[List[str]]):

    tests_to_run = CODE_GEN_TESTS
    if test_name:
        tests_to_run = [test for test in CODE_GEN_TESTS if test.name == test_name]
        if not tests_to_run:
            print(f"No test found with name: {test_name}")
            exit(1)

    if tags:
        tests_to_run = [test for test in tests_to_run if any(tag in tags for tag in test.test_case_core.tags)]
        if not tests_to_run:
            print(f"No tests found with tags: {tags}")
            exit(1)

    print(f"Collected {len(tests_to_run)} tests")

    # Filter prompts if prompt name provided
    print("Collecting prompts...")
    prompt_generators_to_test = CHAT_PROMPT_GENERATORS
    if prompt_name:
        prompt_generators_to_test = [prompt for prompt in CHAT_PROMPT_GENERATORS if prompt.prompt_name == prompt_name]
        if not prompt_generators_to_test:
            print(f"No prompt found with name: {prompt_name}")
            exit(1)
    print(f"Collected {len(prompt_generators_to_test)} prompts")


    # Mapping from prompt name to test results for each prompt we test
    test_case_results: Dict[str, List[TestCaseResult]] = {}
    for prompt_generator in prompt_generators_to_test:
        test_case_results[prompt_generator.prompt_name] = []
        for test in tests_to_run:
            test_case_result = run_code_gen_test(test, prompt_generator)
            test_case_results[prompt_generator.prompt_name].append(test_case_result)

    print_test_case_result_tables(test_case_results)
    

def run_code_gen_test(test: CodeGenTestCase, prompt_generator: ChatPromptGenerator) -> TestCaseResult:
    print(f"Running test: {test.name}")
                
    # Get the script from the cells
    current_cell_contents_script = get_script_from_cells(test.test_case_core.notebook_state.cell_contents)

    # Get the expected code script 
    expected_code = current_cell_contents_script + "\n" + test.test_case_core.expected_code

    # Create the actual code script produced by the LLM
    prompt = prompt_generator.get_prompt(test.user_input, test.test_case_core.notebook_state)
    ai_generated_code = get_open_ai_completion(prompt)
    print(f"AI generated code:\n{ai_generated_code}")
    actual_code = current_cell_contents_script + "\n" + ai_generated_code

    # So that we can compare the results of the two scripts, create global context for 
    # each script. When calling exec, the globals are updated in place.
    expected_globals = {}
    actual_globals = {}

    try:
        exec(expected_code, expected_globals)
        exec(actual_code, actual_globals)
    except Exception as e:
        # Fail early if we can't execute the code
        print("Test Failed: ")
        print(f"Expected code:\n{expected_code}")
        print(f"\nActual code:\n{actual_code}")
        print(f"Error: {e}")
        return TestCaseResult(test=test, passed=False)

    expected_globals = get_globals_to_compare(expected_globals, test.test_case_core.variables_to_compare)
    actual_globals = get_globals_to_compare(actual_globals, test.test_case_core.variables_to_compare)

    return TestCaseResult(test=test, passed=are_globals_equal(expected_globals, actual_globals))
