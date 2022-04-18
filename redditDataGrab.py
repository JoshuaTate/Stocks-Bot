import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import praw
from psaw import PushshiftAPI
import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

def initRedditObj():
    reddit = praw.Reddit(
    client_id="9P1xnNEBwr3_RA",
    client_secret="OFRhsjYcsE5ou5R3kh_OpDSR2sM5vQ",
    user_agent="Agent1",
    username="YetAnotherStonksBot",
    password="password")

    api = PushshiftAPI(reddit)
    return(postDict,api)

def pullPostData(api,start,end,subname):
    gen = api.search_comments(after=start,before=end,
                              subreddit=subname)
    print(gen)

def test():
    mentionsPerDay = {}
    for key in postDict:
        for i in range(len(keywords)):
            if(keywords[i] in postDict[key][1]):
                if(str(postDict[key][0].date()) in mentionsPerDay):
                    mentionsPerDay[str(postDict[key][0].date())] += 1
                else:
                    mentionsPerDay[str(postDict[key][0].date())] = 1
                print(postDict[key][0])
                print(postDict[key][1])
                print(postDict[key][2])
                break
    return(mentionsPerDay)

'''
        all_comments = submission.comments.list()
        for comment in all_comments:
            print(comment.body)
            print(time)
            print(comment.score)
            print(comment.parent_id)
    
    # can also get a particular submission with: submission = reddit.submission(id="123xyz")

    # assume you have a Reddit instance bound to variable `reddit`
    top_level_comments = list(submission.comments)
    all_comments = submission.comments.list()
'''

def get_date(submission):
    time = submission.created
    return datetime.datetime.fromtimestamp(time)

if(__name__=="__main__"):
    time = datetime.datetime(2012, 3, 16, 1, 0)
    print(time)
    #pullPostData(api,start,end,subname)
