import json
import time
import string
import re
import os
from IPython.utils import io
import urllib.request
from datetime import datetime
import openai
from openai import OpenAI
from serpapi import GoogleSearch
from tqdm import tqdm
from config import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from argparse import ArgumentParser, Namespace
import argparse
from utils import *
from pydantic import BaseModel

cache_path = 'data/cache'
os.makedirs(cache_path, exist_ok=True)

NOW_TIME = datetime.now().strftime('%Y-%m-%d-%H-%M')
error_log_file = '{}.log'.format(NOW_TIME)

if not os.path.exists(error_log_file):
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    with open(error_log_file, 'w') as f:
        f.write('')

def my_log(txt):
    with open(error_log_file, 'a') as f:
        f.write(txt + '\n')


def disambiguate(claim):
    prompt = """AMBIGUITY CRITERIA: Ambiguity manifests in diverse forms, including: 
- Similar names denoting distinct entities. 
- Varied interpretations stemming from insufficient information. 
- Multiple understandings arising from vague or unclear information. 

Instructions: 
- Identify the main SUBJECT within the claim. 
- Determine if the SUBJECT is ambiguous according to the provided AMBIGUITY CRITERIA. 
- Utilize your world knowledge to enumerate potential DISAMBIGUATIONS for the identified SUBJECT. 
- Strictly adhere to the context information provided in the claim and avoid introducing DISAMBIGUATION options that are inconsistent with the claim's content. 
- Specify the TYPE of information employed for disambiguation based on the list of DISAMBIGUATIONS. 
- If the SUBJECT does not have ambiguous interpretations, return None 
- Provide an explanation of the method used to arrive at the final response. 

Format your response as a combination of explanation and a dictionary with the following structure: 
##EXPLANATION##: 
<step-by-step-explanations> 
##RESPONSE##: 
{"subject": <subject>, "disambiguations":[ <instance-1>, <instance-2>..], "disambiguation_type": <type>} 

Example 1: 
##CLAIM##: David Heyman, born in 1961 in England, is the founder of Heyday Films. 
##EXPLANATION##: The SUBJECT of the claim is "David Heyman". Based on my world knowledge, there are multiple individuals who share similar names, such as "David Heyman - the British film producer" and "David Heyman - the Chairman of the Board of UK HPA."  To differentiate between them, it is crucial to consider their respective occupations. This criterion offers a  clearer disambiguation compared to nationality, as both individuals are British and thus nationality alone does not  provide sufficient distinguishing information. 
##RESPONSE##: 
{"subject": "David Heyman", "disambiguations": ["David Heyman - British film producer, founder of Heyday Films", "David L. Heyman - Chairman of the Board of UK HPA"], "disambiguation_type": "Occupation"} 

Example 2: 
##CLAIM##: Ruth Bader Ginsburg served as a Supreme Court justice. 
##EXPLANATION##: The SUBJECT is "Ruth Bader Ginsburg". According to my world knowledge, this is a unique individual and I am not aware  of any other individuals/entities with a similar name. Hence, there are no ambiguous interpretations of this SUBJECT and the claim requires no further disambiguation. 
##RESPONSE##: 
{"subject": "Ruth Bader Ginsburg", "disambiguations": "None"} 

Example 3: 
##CLAIM##: Charles Osgood, the american television commentator, is best known for hosting CBS News Sunday Morning. 
##EXPLANATION##: The SUBJECT in focus is "Charles Osgood". Based on my world knowledge, there are two notable individuals with similar names: "Charles Osgood - American radio and television commentator" and "Charles E. Osgood - American psychologist." Given the ambiguity surrounding the name,  specifying the individual's profession serves as an apt disambiguation method. 
##RESPONSE##: 
{"subject": "Charles Osgood", "disambiguations": ["Charles Osgood - American radio and television commentator",  "Charles E. Osgood - American psychologist"], "disambiguation_type": "Profession"} 

Similarly, disambiguate the following claim by detecting the main SUBJECT and disambiguation information for the SUBJECT using your world knowledge.  Generate an EXPLANATION followed by dictionary-formatted RESPONSE. 
##CLAIM##: [claim] 
##EXPLANATION##:""".replace("[claim]", claim)
    return call_gpt(prompt)



def question_answering(claim, subclaims, question, evidence, variable):
    prompt = ("""Assign a value to the variable {answer_1} based on the content in the evidence. Ensure that each subclaim forms a complete and coherent sentence. Only output the value. If no answer can be found in the evidence, output 'unknown'.

##SUBCLAIMS##: 
[subclaims]
##QUESTION##: 
[question]
##EVIDENCE##: 
[evidence]
##ANSWER##:
[variable] =""".replace("[claim]", claim)
              .replace('[subclaims]', "\n".join(subclaims))
              .replace("[question]", question)
              .replace('[evidence]', evidence)
              .replace("[variable]", variable))
    return call_gpt(prompt)

def decompose(claim):
    prompt = """Decompose the following claim.

DECOMPOSITION CRITERIA: 
- Each subclaim should be a single, verifiable fact. Avoid compound statements.
- Subclaims should be independently verifiable, meaning the truth of one subclaim does not depend on another.
- Subclaims should be directly related to the main claim and contribute to its overall verification.
- Subclaims should be specific and detailed enough to allow for precise fact-checking.
- Subclaims should be clear and unambiguous to avoid misinterpretation.
- Ensure that the decomposition is logical and consistent with the original claim.
- Utilize your internal knowledge. If you lack some knowledge, replace it with a variable.
- Generate corresponding questions to verify the corresponding questions. The questions should be easy to retrieve from knowledge base.

Example 1:
##CLAIM##: The Rodney King riots took place in a county in the U.S. with a population of over one million.
##SUBCLAIMS##:
The Rodney King riots took place in {answer_1}.
The population of {answer_1} is {answer_2}.
{answer_2} is over one million.
##QUESTIONS##:
Where did The Rodney King riots take place? {answer_1}
What is the population of {answer_1}? {answer_2}

Example 2:
##CLAIM##: Says 21,000 Wisconsin residents got jobs in 2011, but 18,000 of them were in other states.
##SUBCLAIMS##:
21,000 Wisconsin residents got jobs in 2011.
18,000 Wisconsin residents who got jobs in 2011 were employed in other states.
##QUESTIONS##:
How many Wisconsin residents got jobs in 2011? {answer_1}
How many Wisconsin residents who got jobs in 2011 were employed in other states? {answer_2}


##CLAIM##: """ + claim
    return call_gpt(prompt)

def extract_questions(response):
    """Extract questions from decompose(claim) response."""
    lines = response.splitlines()
    claim = ''
    subclaims = []
    questions = []
    flag_subclaim = False
    flag_questions = False
    for line in lines:
        if line.strip() == '':
            continue
        if line.startswith('##CLAIM##:'):
            claim = line.split('##CLAIM##:')[1]
        if line.startswith('##SUBCLAIMS##:'):
            flag_subclaim = True
            flag_questions = False
            continue
        if line.startswith('##QUESTIONS##:'):
            flag_subclaim = False
            flag_questions = True
            continue
        if flag_subclaim:
            subclaims.append(line)
        if flag_questions:
            questions.append(line)
    return claim, subclaims, questions




def factchecking(claim, questions, subclaims, evidence_list):
    answers_str = "\n".join(['- ' + a for a in evidence_list])
    questions_str = "\n".join(questions)
    subclaims_str = "\n".join(subclaims)
    prompt = ("""Verify the following claim.

Use the following meter to reflect the relative accuracy of the claim. The meter has three ratings, in decreasing level of truthfulness:
- TRUE: This rating indicates that the primary elements of a claim are demonstrably true.
- FALSE: This rating indicates that the primary elements of a claim are demonstrably false.
- MIXTURE: This rating indicates that a claim has significant elements of both truth and falsity to it such that it could not fairly be described by any other rating.

Instructions: 
- Replace the variables in the subclaim with the correct answer based on what is in the EVIDENCE.
- Fact-check each suclaim and label it as a label in the above meter.
- Based on the evidence and your internal knowledge, give the final result for this claim.

Format your response as a dictionary with the following structure: 
{"subclaim_ratings": [{"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}, {"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}], "final_rating": <final rating>, "explanation": <overall explanation>}

Example:
##CLAIM##: The Rodney King riots took place in the most populous county in the USA.
##QUESTIONS##:
Where did The Rodney King riots take place? {answer_1}
##SUBCLAIMS##:
The Rodney King riots took place in {answer_1}.
{answer_1} is the most populous county in the USA.
##EVIDENCE##:
- The riots first began at an intersection in South Los Angeles — Florence and Normandie — according to news reports and firsthand accounts in the ...
- List ; 1, Los Angeles · California, 10,509.87, 4,057.88, 10,014,009 ; 2, Cook · Illinois, 2,448.38, 945.33, 5,275,541 ; 3, Harris · Texas, 4,411.99, 1,703.48 ...
##RESPONSE##: 
{"subclaim_ratings": [{"subclaim": "The Rodney King riots took place in Los Angeles.", "rating": "TRUE", "explanation": "The evidence consistently indicates that the Rodney King riots took place in Los Angeles."}, {"subclaim": "Los Angeles is the most populous county in the USA.", "rating": "TRUE", "explanation": "Los Angeles County is indeed the most populous county in the United States."}], "final_rating": "TRUE", "explanation": "Given that both subclaims are rated as TRUE, the overall claim is also rated as: TRUE"}

##CLAIM##: [claim]
##QUESTIONS##:
[questions]
##SUBCLAIMS##:
[subclaims]
##EVIDENCE##:
[evidence]
##RESPONSE##:""".replace("[claim]", claim)
              .replace("[questions]", questions_str)
              .replace("[subclaims]", subclaims_str).
              replace("[evidence]", answers_str))
    return call_gpt(prompt)



def wikidata_search(query):
    # If there are no results in the cache, call the API to perform the search.
    service_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': query,
        'language': 'en',
        'limit': 20,
        'format': 'json'
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    try:
        if len(response['search']) > 0:
            qid = response['search'][0]['id']
            desc = response['search'][0]['description']
            label = response['search'][0]['label']
            item = {
                'qid': qid,
                'label': label,
                'desc': desc
            }
            return item
    except Exception as e:
        print(response)
        print(str(e))
    return None

def test_RAWFC_open_book(now_time):
    dataset_path = './data/RAWFC/test'

    log_path = f'./log/{now_time}/RAWFC/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    my_log('=== start ' + log_path)

    filelist = [f for f in os.listdir(dataset_path)]
    for ff in tqdm(filelist):
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            to_file = os.path.join(log_path, str(idx) + '.txt')
            if os.path.exists(to_file):
                continue

            try:
                print('---------- idx ----------')
                print(str(idx) + '\n')

                # claim = "Georgia has had ʺmore bank failures than any other state.ʺ"
                resp = decompose(claim)
                _, subclaims, questions = extract_questions(resp)

                # QA
                answers = {}  # e.g.  {'{answer_1}': 'Finland'}
                evidence_list = []
                for q in questions:
                    print("question:", q)
                    query = q
                    flag_searched = False

                    # Extract variables using regular expressions.
                    pattern = r'\{[^}]*\}'
                    matches = re.findall(pattern, q)

                    for m in matches:
                        # If a known answer exists in the question, replace it.
                        if m in answers.keys():
                            query = q.replace(m, answers[m])
                            continue

                        # If there are unknown variables, search for the answers.
                        query = q.replace(m, "")
                        ans = 'unknown'

                        # Use Google to search for the answer. Prioritize the top-ranked search results, and stop checking further evidence once the answer is found.
                        flag_searched = True
                        search_res = get_answer(query)
                        for evidence in search_res:
                            value = question_answering(claim, subclaims, q, evidence, m)
                            print("value:", value)
                            if value != 'unknown':
                                ans = value
                                evidence_list.append(evidence)
                                break
                        answers[m] = ans
                        if ans == 'unknown':
                            evidence_list.append(search_res[0])

                    # If it hasn’t been searched before, it indicates that this is a question without variables.
                    if not flag_searched:
                        search_res = get_answer(query)
                        evidence_list.append(search_res[0])

                print("------- QA -------")
                print("answers:")
                for k, v in answers.items():
                    print(k, v)
                print("evidence:")
                for e in evidence_list:
                    print(e)

                resp2 = factchecking(claim, questions, subclaims, evidence_list)
                with open(to_file, 'w') as f:
                    f.write(resp)
                    f.write('---------------------\n')
                    f.write(json.dumps(answers))
                    f.write('---------------------\n')
                    f.write("\n".join(evidence_list))
                    f.write('---------------------\n')
                    f.write(resp2)
                    f.write('---------------------\n')
                    f.write("ground truth: {}".format(label))
                print("saved to", to_file)
            except Exception as e:
                my_log("-- error: {}, idx: {}".format(str(e), idx))


def eval_RAWFC_open_book(runtime=''):
    dataset_name = 'RAWFC'
    dataset_path = './data/{}/test'.format(dataset_name)

    results_path = 'log/{}/{}/'.format(runtime, dataset_name)

    y_pred = []
    y_true = []
    filelist = [f for f in os.listdir(dataset_path)]
    for ff in tqdm(filelist):
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            res_file = os.path.join(results_path, str(idx) + '.txt')
            if not os.path.exists(res_file):
                continue
            print(res_file)
            pred = read_res(res_file)
            if pred is None:
                print("Error: the pred is None")
                exit()
            print(claim, '\n', label, pred, '\n')
            y_true.append(num_label(label))
            y_pred.append(num_label(pred))

    target_names = ['true', 'mixtrue', 'false']
    print(y_pred)
    print(y_true)
    report_txt = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=False,
                                       zero_division=1)
    print(report_txt)


def test_one_sample(claim):
    resp = decompose(claim)
    _, subclaims, questions = extract_questions(resp)

    # QA
    answers = {}  # e.g.  {'{answer_1}': 'Finland'}
    evidence_list = []
    for q in questions:
        print("question:", q)
        query = q
        flag_searched = False

        # Extract variables using regular expressions.
        pattern = r'\{[^}]*\}'
        matches = re.findall(pattern, q)

        for m in matches:
            # If a known answer exists in the question, replace it.
            if m in answers.keys():
                query = q.replace(m, answers[m])
                continue

            # If there are unknown variables, search for the answers.
            query = q.replace(m, "")
            ans = 'unknown'

            # Use Google to search for the answer. Prioritize the top-ranked search results, and stop checking further evidence once the answer is found.
            flag_searched = True
            search_res = get_answer(query)
            for evidence in search_res:
                value = question_answering(claim, subclaims, q, evidence, m)
                print("value:", value)
                if value != 'unknown':
                    ans = value
                    evidence_list.append(evidence)
                    break
            answers[m] = ans
            if ans == 'unknown':
                evidence_list.append(search_res[0])

        # If it hasn’t been searched before, it indicates that this is a question without variables.
        if not flag_searched:
            search_res = get_answer(query)
            evidence_list.append(search_res[0])

    print("------- QA -------")
    print("answers:")
    for k, v in answers.items():
        print(k, v)
    print("evidence:")
    for e in evidence_list:
        print(e)

    resp2 = factchecking(claim, questions, subclaims, evidence_list)

def read_prompt(file='prompts/FactChecking.md'):
    with open(file, 'r') as f:
        content = f.read()

    # Define regular expressions to match the content from System and User.
    system_pattern = r"System:\n(.*?)\nUser:"
    user_pattern = r"User:\n(.*?)$"

    # Use re.DOTALL to allow matching newline characters.
    system_match = re.search(system_pattern, content, re.DOTALL)
    user_match = re.search(user_pattern, content, re.DOTALL)

    # Extract the content; if no match is found, set the result to None.
    system_text = system_match.group(1).strip() if system_match else None
    user_text = user_match.group(1).strip() if user_match else None

    return system_text, user_text

class FCResponse(BaseModel):
    Label: str
    Steps: list[str]


def factchecking_RAWFC(claim, evidence_list, hparams=None):
    if hparams.prompt_mode == 'Markdown':
        evidence_str = "\n".join(['- ' + a for a in evidence_list])
        system_text, user_text = read_prompt('prompts/FactChecking.md')
        user_text = user_text.replace("[Claim]", claim).replace("[Evidence]", evidence_str)
        event = call_gpt(system_text, user_text)
    elif hparams.prompt_mode == 'JSON':
        evidence_str = json.dumps(evidence_list, indent=4)
        system_text, user_text = read_prompt('prompts/FactChecking.json')
        user_text = user_text.replace("[Claim]", claim).replace("[Evidence]", evidence_str)
        event = call_gpt(system_text, user_text)
    elif hparams.prompt_mode == 'YAML':
        evidence_str = "\n".join(['- ' + a for a in evidence_list])
        system_text, user_text = read_prompt('prompts/FactChecking.yaml')
        user_text = user_text.replace("[Claim]", claim).replace("[Evidence]", evidence_str)
        event = call_gpt(system_text, user_text)
    else: # Plaintext
        evidence_str = "\n".join(evidence_list)
        system_text, user_text = read_prompt('prompts/FactChecking.txt')
        user_text = user_text.replace("[Claim]", claim).replace("[Evidence]", evidence_str)
        event = call_gpt(system_text, user_text)

    return event, system_text, user_text

def test_RAWFC_gold(now_time, hparams):
    dataset_path = './data/RAWFC/test'

    log_path = f'./log/{now_time}/RAWFC/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    my_log('=== start ' + log_path)

    gold_evidence_length = []

    filelist = [f for f in os.listdir(dataset_path)]
    for ff in tqdm(filelist):
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            to_file = os.path.join(log_path, str(idx) + '.txt')
            if os.path.exists(to_file):
                continue

            # Gold Evidence
            evidence_list = []
            for report in item['reports']:
                for s in report['tokenized']:
                    if s['is_evidence'] > 0 and len(evidence_list) < 40:
                        evidence_list.append(s['sent'])
            # gold_evidence_length.append(len(evidence_list))

            try:
                print('---------- idx ----------')
                print(str(idx) + '\n')

                # claim = "Georgia has had ʺmore bank failures than any other state.ʺ"
                resp = decompose(claim)
                _, subclaims, questions = extract_questions(resp)

                resp2 = factchecking_RAWFC(claim, questions, subclaims, evidence_list)
                with open(to_file, 'w') as f:
                    f.write(resp)
                    f.write('---------------------\n')
                    f.write(resp2)
                    f.write('---------------------\n')
                    f.write("ground truth: {}".format(label))
                print("saved to", to_file)
            except Exception as e:
                my_log("-- error: {}, idx: {}".format(str(e), idx))



def eval_RAWFC_gold(runtime='', hparams=None):
    dataset_name = 'RAWFC'
    dataset_path = './data/{}/test'.format(dataset_name)

    results_path = 'log/{}/{}/'.format(runtime, dataset_name)

    y_pred = []
    y_true = []
    filelist = [f for f in os.listdir(dataset_path)]
    for ff in filelist:
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            res_file = os.path.join(results_path, str(idx) + '.txt')
            if not os.path.exists(res_file):
                continue
            print(res_file)
            pred = read_res(res_file)
            if pred is None:
                print("Error: the pred is None")
                exit()
            print(claim, '\n', label, pred, '\n')
            y_true.append(num_label(label))
            y_pred.append(num_label(pred))

    target_names = ['true', 'mixtrue', 'false']
    print(y_pred)
    print(y_true)
    report_txt = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=False,
                                       zero_division=1)
    print(report_txt)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description="D2F",
        add_help=True,
    )
    parser.add_argument(
        "--now_time",
        default="",
        type=str,
        required=False,
        help="now_time"
    )
    parser.add_argument(
        "--data_set",
        default="RAWFC",
        type=str,
        choices=["LIAR_RAW", "RAWFC"],
        required=False,
        help="data_set: [LIAR_RAW, RAWFC]"
    )
    parser.add_argument(
        "--experiment_name",
        default="D2F",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--evidence_mode",
        default="Search",
        type=str,
        choices=["Internal", "Gold", "KP", "Search"],
        required=False,
        help="evidence_mode: [Internal, Gold, KP]"
    )
    parser.add_argument(
        "--prompt_mode",
        default="Markdown",
        type=str,
        choices=["Markdown", "JSON", "YAML", "PlainText"],
        required=False,
        help="prompt_mode: [Markdown, JSON, YAML, PlainText]"
    )

    hparams = parser.parse_args()

    hparams.cache_path = "data/google_search_cache"
    os.makedirs(hparams.cache_path, exist_ok=True)


    if hparams.experiment_name == '':
        hparams.experiment_name = "D2F"
    if hparams.now_time == "":
        hparams.now_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

    hparams.log_path = os.path.join(f'log/{hparams.experiment_name}/{hparams.now_time}_{hparams.data_set}/')
    os.makedirs(hparams.log_path, exist_ok=True)

    if hparams.data_set == 'RAWFC':
        test_RAWFC_gold(hparams.now_time, hparams)
        eval_RAWFC_gold(hparams.now_time, hparams)

    print("done")
