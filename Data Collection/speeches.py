import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

def get_recent_speeches():
    """
    This function retrieves speeches on or after 2006 from FED website and return a dictionary
    """

    result = requests.get("https://www.federalreserve.gov/json/ne-speeches.json")

    #check if successfully retreived the website
    if result.status_code != 200:
        raise Exception("Cannot connet to FED website.")

    result = result.content.decode("utf-8-sig") #decode json according to the websites encoding

    data = {"Date":[], "Speaker": [], "Title": [], "Content": [], "Location": [] }

    speeches = json.loads(result) #parse the result as json
    for speech in speeches:
        if "d" not in speech.keys(): #escape from some non-speech data
            continue

        content_result = requests.get("https://www.federalreserve.gov/"+speech['l']) #retreive the website for the content

        #check if successfully retreived the website
        if content_result.status_code != 200:
            print(f"Error retrieving {speech['l']}")
            continue

        #Search for the content using BeautifulSoup
        soup = BeautifulSoup(content_result.content,"html.parser")
        article = soup("div",{'id':'article'})[0]
        article = article.find_all("div", recursive=False)[2]
        content = [x.get_text() for x in article.find_all('p')]

        #Store the content
        data["Date"].append(speech['d'].split(" ")[0])
        data["Speaker"].append(speech['s'])
        data["Title"].append(speech['t'])
        data["Location"].append(speech['lo'])
        data['Content'].append(content)

    return data

def get_older_speeches():
    """
    This function retrieves speeches before 2006 from FED website and return a dictionary
    """
    years = list(range(1996,2006))
    data = {"Date":[], "Speaker": [], "Title": [], "Content": [], "Location": [] }

    for yr in years:
        result = requests.get(f"https://www.federalreserve.gov/newsevents/speech/{yr}speech.htm")

        #check if successfully retreived the website
        if result.status_code != 200:
            raise Exception("Cannot connet to FED website.")

        #get the list of speeches
        soup = BeautifulSoup(result.content,"html.parser")
        speeches = soup("ul",{'id':'speechIndex'})[0]
        speeches = speeches.find_all('li')

        for speech in speeches:
            #get link for content and title
            title = speech.find('a')
            link = title['href']
            if link[0] != "\\":
                link = "\\"+link

            content_result = requests.get(f"https://www.federalreserve.gov{link}")
            
            #check if successfully retreived the website
            if content_result.status_code != 200:
                print(f"Error retrieving {link}")
                continue
            
            #get date
            data['Date'].append(speech.contents[0].strip())
            
            #get speaker
            speaker = speech.find('div',{'class':'speaker'})
            data['Speaker'].append(speaker.text.strip())

            #get location
            location = speech.find('div',{'class':'location'})
            data['Location'].append(location.text.strip())

            #store title
            data['Title'].append(title.text.strip())

            #get content
            soup = BeautifulSoup(content_result.content,"html.parser")
            article = soup("p") #contents are stored as text
            content = [x.find(text=True, recursive=False) for x in article]
            data["Content"].append(content)

    return data

if __name__ == '__main__':
    # Get recent speeches (after 2006)
    df_speeches = pd.DataFrame(data=get_recent_speeches())
    df_speeches["Date"] = pd.to_datetime(df_speeches["Date"])

    # Get older speeches (before 2006)
    df_speeches2 = pd.DataFrame(data=get_older_speeches())
    df_speeches2["Date"] = pd.to_datetime(df_speeches2["Date"])

    # # combine the data and save to csv
    df_speeches = pd.concat([df_speeches,df_speeches2],ignore_index=True)
    df_speeches.to_csv("../Data/speeches.csv",index=False)