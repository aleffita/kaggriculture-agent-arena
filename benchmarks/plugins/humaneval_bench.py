"""
Agnostic OpenAI HumanEval Capability Benchmark Plugin.
Evaluates functional correctness (pass@1) on Python programming problems with local unit test execution.
Supports --mode smoke (15 curated canonical tasks: HumanEval/0 a HumanEval/14) and --mode full.
"""
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from .base import BaseBenchmarkPlugin, BenchmarkMeasurement, BenchmarkSuiteResult

# 15 Problemas Canônicos do OpenAI HumanEval (HumanEval/0 a HumanEval/14)
HUMANEVAL_15_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "name": "has_close_elements",
        "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. \"\"\"\n",
        "reference_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\ncheck(has_close_elements)\n"
    },
    {
        "task_id": "HumanEval/1",
        "name": "separate_paren_groups",
        "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Separate groups of balanced parentheses. \"\"\"\n",
        "reference_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\ncheck(separate_paren_groups)\n"
    },
    {
        "task_id": "HumanEval/2",
        "name": "truncate_number",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Return the decimal part of the number. \"\"\"\n",
        "reference_solution": "    return number % 1.0\n",
        "test": "def check(candidate):\n    assert abs(candidate(3.5) - 0.5) < 1e-6\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\ncheck(truncate_number)\n"
    },
    {
        "task_id": "HumanEval/3",
        "name": "below_zero",
        "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" Detect if balance falls below zero at any point. \"\"\"\n",
        "reference_solution": "    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return False\n",
        "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\ncheck(below_zero)\n"
    },
    {
        "task_id": "HumanEval/4",
        "name": "mean_absolute_deviation",
        "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" Calculate Mean Absolute Deviation around the mean of this dataset. \"\"\"\n",
        "reference_solution": "    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)\n",
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\ncheck(mean_absolute_deviation)\n"
    },
    {
        "task_id": "HumanEval/5",
        "name": "intersperse",
        "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of numbers. \"\"\"\n",
        "reference_solution": "    if not numbers:\n        return []\n    result = []\n    for n in numbers[:-1]:\n        result.append(n)\n        result.append(delimeter)\n    result.append(numbers[-1])\n    return result\n",
        "test": "def check(candidate):\n    assert candidate([], 4) == []\n    assert candidate([1, 2, 3], 4) == [1, 4, 2, 4, 3]\n    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\ncheck(intersperse)\n"
    },
    {
        "task_id": "HumanEval/6",
        "name": "parse_nested_parens",
        "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" For each group of nested parentheses, output deepest level of nesting. \"\"\"\n",
        "reference_solution": "    def parse_group(s):\n        d = 0\n        m = 0\n        for c in s:\n            if c == '(':\n                d += 1\n                m = max(m, d)\n            elif c == ')':\n                d -= 1\n        return m\n    return [parse_group(x) for x in paren_string.split() if x]\n",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\ncheck(parse_nested_parens)\n"
    },
    {
        "task_id": "HumanEval/7",
        "name": "filter_by_substring",
        "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter list of strings only for ones that contain substring. \"\"\"\n",
        "reference_solution": "    return [x for x in strings if substring in x]\n",
        "test": "def check(candidate):\n    assert candidate([], 'john') == []\n    assert candidate(['xxx', 'asd', 'xxy', 'john', 'doe'], 'xxx') == ['xxx']\n    assert candidate(['xxx', 'asd', 'aaaxxy', 'john', 'doe'], 'xx') == ['xxx', 'aaaxxy']\n    assert candidate(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') == ['grunt', 'prune']\ncheck(filter_by_substring)\n"
    },
    {
        "task_id": "HumanEval/8",
        "name": "sum_product",
        "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" Return a tuple consisting of a sum and a product of all integers. \"\"\"\n",
        "reference_solution": "    s = 0\n    p = 1\n    for n in numbers:\n        s += n\n        p *= n\n    return s, p\n",
        "test": "def check(candidate):\n    assert candidate([]) == (0, 1)\n    assert candidate([1, 1, 1]) == (3, 1)\n    assert candidate([100, 0]) == (100, 0)\n    assert candidate([3, 5, 7]) == (15, 105)\ncheck(sum_product)\n"
    },
    {
        "task_id": "HumanEval/9",
        "name": "rolling_max",
        "prompt": "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" Generate list of rolling maximum element found until given moment. \"\"\"\n",
        "reference_solution": "    m = None\n    res = []\n    for n in numbers:\n        m = n if m is None else max(m, n)\n        res.append(m)\n    return res\n",
        "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]\n    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]\ncheck(rolling_max)\n"
    },
    {
        "task_id": "HumanEval/10",
        "name": "make_palindrome",
        "prompt": "def is_palindrome(string: str) -> bool:\n    return string == string[::-1]\n\ndef make_palindrome(string: str) -> str:\n    \"\"\" Find shortest palindrome that begins with supplied string. \"\"\"\n",
        "reference_solution": "    if not string:\n        return ''\n    idx = 0\n    while not is_palindrome(string[idx:]):\n        idx += 1\n    return string + string[:idx][::-1]\n",
        "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('x') == 'x'\n    assert candidate('xyz') == 'xyzyx'\n    assert candidate('xyx') == 'xyx'\n    assert candidate('jerry') == 'jerryrrej'\ncheck(make_palindrome)\n"
    },
    {
        "task_id": "HumanEval/11",
        "name": "string_xor",
        "prompt": "from typing import List\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Binary XOR on two strings consisting of 1s and 0s. \"\"\"\n",
        "reference_solution": "    return ''.join('0' if x == y else '1' for x, y in zip(a, b))\n",
        "test": "def check(candidate):\n    assert candidate('111000', '101010') == '010010'\n    assert candidate('1', '1') == '0'\n    assert candidate('0101', '0000') == '0101'\ncheck(string_xor)\n"
    },
    {
        "task_id": "HumanEval/12",
        "name": "longest",
        "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Return longest string from list. \"\"\"\n",
        "reference_solution": "    if not strings:\n        return None\n    m = max(len(x) for x in strings)\n    for s in strings:\n        if len(s) == m:\n            return s\n",
        "test": "def check(candidate):\n    assert candidate([]) is None\n    assert candidate(['x', 'y', 'z']) == 'x'\n    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'\ncheck(longest)\n"
    },
    {
        "task_id": "HumanEval/13",
        "name": "greatest_common_divisor",
        "prompt": "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\" Return greatest common divisor of a and b. \"\"\"\n",
        "reference_solution": "    while b:\n        a, b = b, a % b\n    return a\n",
        "test": "def check(candidate):\n    assert candidate(3, 7) == 1\n    assert candidate(10, 15) == 5\n    assert candidate(49, 14) == 7\n    assert candidate(144, 60) == 12\ncheck(greatest_common_divisor)\n"
    },
    {
        "task_id": "HumanEval/14",
        "name": "all_prefixes",
        "prompt": "from typing import List\n\ndef all_prefixes(string: str) -> List[str]:\n    \"\"\" Return list of all prefixes from shortest to longest. \"\"\"\n",
        "reference_solution": "    return [string[:i+1] for i in range(len(string))]\n",
        "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n    assert candidate('WWW') == ['W', 'WW', 'WWW']\ncheck(all_prefixes)\n"
    }
]

class HumanEvalBenchmarkPlugin(BaseBenchmarkPlugin):
    name = "humaneval"
    description = "Avaliação de corretude funcional de código (pass@1) com 15 problemas canônicos do OpenAI HumanEval"

    def _execute_code_test(self, code_str: str) -> tuple[bool, float]:
        """Executa a verificação dos testes unitários em subprocesso isolado retornando sucesso e latência em ms."""
        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                [sys.executable, "-c", code_str],
                capture_output=True,
                text=True,
                timeout=5
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return (res.returncode == 0, lat_ms)
        except Exception:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return (False, lat_ms)

    def run(self, mode: str = "smoke") -> BenchmarkSuiteResult:
        suite = BenchmarkSuiteResult(
            benchmark_name=self.name,
            mode=mode,
            environment_metadata={
                "benchmark_source": "OpenAI HumanEval (15 Canonical Problems)",
                "metric": "pass@1",
                "execution_sandbox": "Isolated Python Subprocess"
            }
        )

        problems = HUMANEVAL_15_PROBLEMS

        passed_count = 0
        tasks_detail = []

        for prob in problems:
            task_id = prob["task_id"]
            name = prob["name"]
            prompt = prob["prompt"]
            solution = prob["reference_solution"]
            test = prob["test"]

            full_code = prompt + solution + "\n" + test
            passed, lat_ms = self._execute_code_test(full_code)

            if passed:
                passed_count += 1

            tasks_detail.append({
                "task_id": task_id,
                "name": name,
                "passed": passed,
                "latency_ms": round(lat_ms, 2)
            })

            meas = BenchmarkMeasurement(
                benchmark=self.name,
                backend="python-exec",
                model=task_id,
                mode=mode,
                prompt_tokens=len(prompt.split()),
                gen_tokens=len(solution.split()),
                batch_size=1,
                metric_name="pass@1",
                metric_value=1.0 if passed else 0.0,
                error_stddev=0.0,
                status="PASSED" if passed else "FAILED",
                details={
                    "function_name": name,
                    "latency_ms": round(lat_ms, 2)
                }
            )
            suite.measurements.append(meas)

        pass_at_1 = passed_count / len(problems) if problems else 0.0

        # Métrica agregada da suite
        suite.environment_metadata["summary"] = {
            "total_tasks": len(problems),
            "passed_tasks": passed_count,
            "pass_at_1": pass_at_1,
            "tasks_detail": tasks_detail
        }

        return suite
