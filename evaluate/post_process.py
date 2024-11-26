import json
import os
import re
import pdb

def build_test_method_for_apps(input_output, test_case_limit = 5):
    test = "def check(candidate):\n"
    for idx, (input, output) in enumerate(zip(input_output['inputs'], input_output['outputs'])):
        if idx >= test_case_limit:
            break
        try:
            test += "\tassert candidate(%r) == %r \n" % (input.strip(), output.strip())
        except:
            test += "\tassert candidate(%s) == %s \n" % (input, output)
    return test

def truncate(d):
    d = d.split('\n\n')
    s = d[0] + '\n\n'
    if len(d)>1:
        for i in d[1:]:
            if 'def' not in i and i and '__main__' not in i:
                s += i + '\n\n'
            else:
                break
    return s

def minimum_indent(lines):
    m_indent = 100
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if indent > 0 and indent < m_indent:
            m_indent = indent
    return m_indent

def check_overall_indent(lines):
    def check_indent(lines):
        for line in lines:
            if "def" not in line and "print" not in line and "__name__" not in line and line[0] != '#' and len(line) - len(line.lstrip()) == 0:
                return True
        return False
    m_indent = minimum_indent(lines) 
    if len(lines) <= 1:
        return False
    elif len(lines[0]) - len(lines[0].lstrip()) == 0:
        if lines[0].strip()[-1] == ':':
            space_num = len(lines[1]) - len(lines[1].lstrip())
            if space_num == m_indent:
                return True
        elif check_indent(lines[1:]):
            return True
    return False


def post_process_code(prompt, code, func_name, m_indent):
    assert type(code) == str
    # truncate
    if f"def {func_name}(" in code:
        return code
    truncation = truncate(code).replace('\r', '\n')
    truncation = re.sub('\n+', '\n', truncation)
    lines = truncation.split('\n')

    # lines = list(filter(lambda x: x.strip() != "" and func_name not in x, lines))
    lines = list(filter(lambda x: x.strip() != "", lines))
    
    lines = list(map(lambda x: x.replace('\t', m_indent), lines))

    if len(lines) == 0:
        pass
    else:
        if check_overall_indent(lines):
            for i in range(len(lines)):
                lines[i] = m_indent + lines[i] 
        elif len(lines[0]) - len(lines[0].lstrip()) == 0:
            lines[0] = m_indent + lines[0]
        else:
            pass
    return prompt.replace('\t', m_indent)+'\n'.join(lines)

def build_test_method_for_CodeForces(input_output):
    test_method = "def check(candidate):\n"
    for idx, (input, output) in enumerate(zip(input_output['inputs'], input_output['outputs'])):
        try:
            test_method += "\tassert candidate(%r) == %r \n" % (input.strip(), output.strip())
        except:
            test_method += "\tassert candidate(%s) == %s \n" % (input, output)
    return test_method

def post_process_code_for_CodeForces(prompt, code, func_name, m_indent):
    assert type(code) == str
    if f"def {func_name}(" in code:
        return code
    truncation = truncate(code).replace('\r', '\n')
    truncation = re.sub('\n+', '\n', truncation)
    lines = truncation.split('\n')
    lines = list(filter(lambda x: x.strip() != "", lines))
    lines = list(map(lambda x: x.replace('\t', m_indent), lines))

    if len(lines) == 0:
        pass
    else:
        if check_overall_indent(lines):
            for i in range(len(lines)):
                lines[i] = m_indent + lines[i] 
        elif len(lines[0]) - len(lines[0].lstrip()) == 0:
            lines[0] = m_indent + lines[0]
        else:
            pass
    return prompt.replace('\t', m_indent)+'\n'.join(lines)
