from config import *
import openai
from openai import OpenAI
from googleapiclient.discovery import build # pip install google-api-python-client
import hashlib
import json
import requests
import os
from IPython.utils import io
from serpapi import GoogleSearch

openai.api_key = OPENAI_API_KEY
serpapi_key = SERPAPI_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

def call_gpt(cur_prompt, stop=["\n"], model = 'gpt-4o-mini-2024-07-18'):
    reasoner_messages = [{"role": "user", "content": cur_prompt}]
    completion = client.beta.chat.completions.parse(
        # model="gpt-3.5-turbo",
        # model="gpt-4o-2024-05-13",
        # model="gpt-4o-mini-2024-07-18",
        model=model,
        messages=reasoner_messages,
        # response_format=FCResponse,
        # stop=stop
    )
    # returned = completion['choices'][0]["message"]["content"]
    # returned = completion.choices[0].message.parsed
    returned = completion.choices[0].message.content
    print("-------- returned start --------")
    print(returned)
    print("-------- returned end   --------")
    return returned


def call_gpt_old_version(cur_prompt, stop=["\n"]):
    reasoner_messages = [{"role": "user", "content": cur_prompt}]
    print("-------- call gpt --------")
    # print(reasoner_messages)
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4o-2024-05-13",
        model="gpt-4o-mini-2024-07-18",
        messages=reasoner_messages,
        # stop=stop
    )
    returned = completion['choices'][0]["message"]["content"]
    print("-------- returned start --------")
    print(returned)
    print("-------- returned end   --------")
    return returned

def num_label(label):
    l = label.strip().lower()
    d = {
        'true': 0,
        'TRUE': 0,

        'half-true': 1,
        'HALF TRUE': 1,
        'MIXTURE': 1,
        'mixture': 1,
        'half': 1,
        'half true': 1,

        'false': 2,
        'FALSE': 2,

        'mostly-true': 0,
        'MOSTLY TRUE': 0,
        'mostly true': 0,

        'barely-true': 1,

        'MOSTLY FALSE': 2,
        'mostly false': 2,

        'pants-fire': 2,
        'pants on fire': 2,
        'PANTS ON FIRE': 2
    }
    if l not in d:
        return 2
    return d[l]



def google_search(query, num_results=10, hparams=None):
    """
    Use the Google Custom Search API to perform the search, and add a caching feature.

    Parameters:
        query: str, the search keywords
        num_results: int, number of search results to return (default is 10, maximum supported is 10)

    Returns:
        •	results: list, a list of search result titles and links
    """
    cache_key = generate_cache_key(query)
    to_file = os.path.join(hparams.cache_path, "{}.json".format(cache_key))

    if os.path.exists(to_file):
        data = json.load(open(to_file))
    else:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": num_results,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            resp = response.json()
            data = {
                'query': query,
                'response': resp,
            }
            with open(to_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("saved to {}".format(to_file))
        except requests.exceptions.RequestException as e:
            print(f"failed: {e}")
            return []
        except KeyError as e:
            print(f"error: {e}")
            return []

    results = []
    for item in data['response'].get("items", []):
        if 'fact' not in item.get("link"):
            results.append(item.get("snippet"))
        # results.append({
        #     "title": item.get("title"),
        #     "link": item.get("link"),
        #     "snippet": item.get("snippet"),
        # })
    return results


def generate_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()


def google_search_with_cache(query, num_results=10):
    cache_key = generate_cache_key(query)
    to_file = os.path.join('./data/google_search_cache', "{}.json".format(cache_key))

    if os.path.exists(to_file):
        data = json.load(open(to_file))
    else:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": num_results,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            resp = response.json()
            data = {
                'query': query,
                'response': resp,
            }
            with open(to_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("saved to {}".format(to_file))
        except requests.exceptions.RequestException as e:
            print(f"failed: {e}")
            return []
        except KeyError as e:
            print(f"error: {e}")
            return []

    results = []
    for item in data['response'].get("items", []):
        if 'fact' not in item.get("link"):
            # results.append(item.get("snippet"))
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })
    return results


def get_answer_serp(question):
    from serpapi import GoogleSearch
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "q": question,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    with io.capture_output() as captured:  # disables prints from GoogleSearch
        search = GoogleSearch(params)
        res = search.get_dict()

    # Use Google to search for evidence. Filter out fact-checking websites.
    if "organic_results" in res.keys():
        for idx in range(len(res["organic_results"])):
            if 'snippet' in res["organic_results"][idx].keys():
                if 'fact' not in res["organic_results"][idx]['link']:
                    toret = res["organic_results"][idx]['snippet']
                    break
            if (idx + 1) == len(res["organic_results"]):
                toret = None
    else:
        toret = None
    return toret

def get_answer(question):
    results = google_search_with_cache(question)
    # for result in results:
    #     print(result)
    if results:
        return results[0]['snippet']
    return None



def call_gpt(cur_prompt, stop=["\n"]):
    reasoner_messages = [{"role": "user", "content": cur_prompt}]
    print("-------- call gpt --------")
    print(reasoner_messages)
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4o-2024-05-13",
        messages=reasoner_messages,
        # stop=stop
    )
    returned = completion['choices'][0]["message"]["content"]
    print("-------- returned start --------")
    print(returned)
    print("-------- returned end   --------")
    return returned


def serp_search(query):
    params = {
        "api_key": serpapi_key,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    res = None
    with io.capture_output() as captured:  # disables prints from GoogleSearch
        search = GoogleSearch(params)
        res = search.get_dict()
    return res

def is_social_media(link):
    sites = ['mobile.twitter.com',
        'twitter.com',
        'x.com',
        'toolbox.google.com',
        'reddit.com',
        'snopes.com',
        'facebook.com',
        'instagram.com',
        'linkedin.com',
        'pinterest.com',
        'snapchat.com',
        'tumblr.com',
        'tiktok.com',
        'youtube.com',
        'vimeo.com',
        'whatsapp.com',
        'quora.com',
        'weibo.com',
        'yelp.com',
        'sina.com.cn',
        'snopes.com',
        'politifact.com',
        'www.truthorfiction.com',
        'www.factcheck.org',
        'fullfact.org',
        'apnews.com',
        'uk.reuters.com',
    ]
    for site in sites:
        if site in link:
            return True
    return False

# def read_res(res_file):
#     with open(res_file, 'r') as f:
#         text = f.read()
#     arr = text.split("---------------------")
#     json_str = arr[3]
#     if json_str.find('```json') > -1:
#         json_str = json_str.replace("```json", "").replace('```', '')
#     pred = json.loads(json_str)['final_rating']
#     return pred

def read_res(res_file):
    with open(res_file, 'r') as f:
        text = f.read()
    # arr = text.split("---------------------")
    # json_str = arr[-2]
    # if json_str.find('```json') > -1:
    #     json_str = json_str.replace("```json", "").replace('```', '')
    # pred = json.loads(json_str)['final_rating']
    pred = text.split("\n")[0].strip()
    return pred


def statistic(data):
    import numpy as np
    import matplotlib.pyplot as plt

    max_value = np.max(data)
    min_value = np.min(data)
    mean_value = np.mean(data)
    median_value = np.median(data)

    print(f"Maximum Value: {max_value}")
    print(f"Minimum Value: {min_value}")
    print(f"Mean Value: {mean_value}")
    print(f"Median Value: {median_value}")

    plt.hist(data, bins=10, edgecolor='black')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()