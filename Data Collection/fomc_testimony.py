import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

def get_recent_testimonies():
  try:
    text = requests.get("https://www.federalreserve.gov/json/ne-testimony.json")
    text.raise_for_status()
    text = text.content.decode("utf-8-sig")
    testimonies = json.loads(text)
    lst_d = []
    lst_s = []
    lst_s1 = []
    for testimony in testimonies:
      if ('2006' in testimony['d']) or ('2007' in testimony['d']) or ('2008' in testimony['d']) or ('2009' in testimony['d']) or ('2010' in testimony['d']) \
      or ('2011' in testimony['d']) or ('2012' in testimony['d']) or ('2013' in testimony['d']) or('2014' in testimony['d']) or ('2015' in testimony['d']) \
      or ('2016' in testimony['d']) or ('2017' in testimony['d']) or ('2018' in testimony['d']) or ('2019' in testimony['d']) or ('2020' in testimony['d']) \
      or ('2021' in testimony['d']) or ('2022' in testimony['d']) or ('2023' in testimony['d']):
        lst_d.append(testimony['d'])#[:testimony['d'].find(" ")])
        lst_s.append(testimony['s'])
        lst_s1.append(testimony['s'])
  except:
      pass

  ser1 = pd.Series(lst_d)
  ser1.loc[68] = '2/15/2006'
  ser1.loc[69] = '3/1/2006'
  ser1 = pd.to_datetime(ser1)
  ser1 = ser1.dt.strftime('%Y%m%d').astype(str)
  ser1.loc[194] = '20110812'
  ser1.loc[230] = '20140114'
  ser1.loc[215] = '20120628'
  ser1.loc[41] = '20180226'
  ser1.loc[52] = '20100323'
  ser1.loc[148] = '20090521'
  lst_d = list(ser1.values)
  ser2 = pd.Series(lst_s)

  for idx,val in enumerate(ser2):
    if val == 'Governor Daniel K. Tarullo':
      ser2.loc[idx] = 'tarullo'
    elif val == 'Kevin M. Bertsch, Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'bertsch'
    elif val == 'Chairman Ben S. Bernanke':
      ser2.loc[idx] = 'bernanke'
    elif val == 'Steven B. Kamin, Director, Division of International Finance':
      ser2.loc[idx] = 'kamin'
    elif val == 'Governor Elizabeth A. Duke':
      ser2.loc[idx] = 'duke'
    elif val == 'Suzanne G. Killian, Senior Associate Director, Division of Consumer and Community Affairs':
      ser2.loc[idx] = 'killian'
    elif val == 'Sandra F. Braunstein, Director, Division of Consumer and Community Affairs':
      ser2.loc[idx] = 'braunstein'
    elif val == 'Michael S. Gibson, Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'gibson'
    elif val == 'Scott G. Alvarez, General Counsel':
      ser2.loc[idx] = 'alvarez'
    elif val == 'Stephanie Martin, Associate General Counsel':
      ser2.loc[idx] = 'martin'
    elif val == 'Matthew J. Eichner, Deputy Director, Division of Research and Statistics':
      ser2.loc[idx] = 'eichner'
    elif val == 'Governor Jerome H. Powell':
      ser2.loc[idx] = 'powell'
    elif val == 'Richard M. Ashton, Deputy General Counsel':
      ser2.loc[idx] = 'ashton'
    elif val == 'Todd Vermilyea, Senior Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'vermilyea'
    elif val == 'Chair Janet L. Yellen':
      ser2.loc[idx] = 'yellen'
    elif val == 'Louise L. Roseman, Director, Division of Reserve Bank Operations and Payment Systems':
      ser2.loc[idx] = 'roseman'
    elif val == 'Maryann Hunter, Deputy Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'hunter'
    elif val == 'Thomas Sullivan, Senior Adviser, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'sullivan'
    elif val == 'Maryann F. Hunter, Deputy Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'hunter'
    elif val == 'Mark E. Van Der Weide, Deputy Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'vanderweide'
    elif val == 'Thomas Sullivan, Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'sullivan'
    elif val == 'Patrick M. Parkinson, Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'parkinson'
    elif val == 'Governor Sarah Bloom Raskin':
      ser2.loc[idx] = 'raskin'
    elif val == 'William R. Nelson, Deputy Director, Division of Monetary Affairs':
      ser2.loc[idx] = 'nelson'
    elif val == 'Chiarman Ben S. Bernanke':
      ser2.loc[idx] = 'bernanke'
    elif val == 'Michael R. Foley, Senior Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'foley'
    elif val == 'Mark E. Van Der Weide, Senior Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'vanderweide'
    elif val == 'Statement for the Record':
      ser2.loc[idx] = 'statement'
    elif val == 'Scott G. Alvarez, General Counsel, and Thomas C. Baxter Jr., General Counsel, Federal Reserve Bank of New York ':
      ser2.loc[idx] = 'alvarez'
    elif val == 'J. Nellie Liang, Director, Office of Financial Stability Policy and Research':
      ser2.loc[idx] = 'liang'
    elif val == 'Vice Chair for Supervision Michael S. Barr':
      ser2.loc[idx] = 'barr'
    elif val == 'Chair Jerome H. Powell':
      ser2.loc[idx] = 'powell'
    elif val == 'Vice Chair Lael Brainard':
      ser2.loc[idx] = 'brainard'
    elif val == 'Vice Chair for Supervision Randal K. Quarles':
      ser2.loc[idx] = 'quarles'
    elif val == 'Mark Van Der Weide, General Counsel':
      ser2.loc[idx] = 'vanderweide'
    elif val == 'Kent Hiteshew, Deputy Associate Director, Division of Financial Stability':
      ser2.loc[idx] = 'hiteshew'
    elif val == 'Testimony by Sheila Clark, Program Director, Office of Diversity and Inclusion':
      ser2.loc[idx] = 'clark'
    elif val == 'Arthur Lindo, Deputy Director, Division of Supervision and Regulation':
      ser2.loc[idx] = 'lindo'
    elif val == 'Esther George, President, Federal Reserve Bank of Kansas City':
      ser2.loc[idx] = 'george'
    elif val == 'Thomas Sullivan, Associate Director, Division of Supervision and Regulation':
      ser2.loc[idx] = 'sullivan'
    elif val == 'Governor Michelle W. Bowman':
      ser2.loc[idx] = 'bowman'
    elif val == 'Vice Chairman for Supervision Randal K. Quarles':
      ser2.loc[idx] = 'quarles'
    elif val == 'Chairman Jerome H. Powell':
      ser2.loc[idx] = 'powell'
    elif val == 'Governor Lael Brainard':
      ser2.loc[idx] = 'brainard'
    elif val == 'Chair Pro Tempore Jerome H. Powell ':
      ser2.loc[idx] = 'powell'
    elif val == 'Jon D. Greenlee, Associate Director, Division of Banking':
      ser2.loc[idx] = 'greenlee'
    elif val == 'Jon D. Greenlee, Associate Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'greenlee'
    elif val == 'Governor Donald L. Kohn':
      ser2.loc[idx] = 'kohn'
    elif val == 'Roger T. Cole, Director, Division of Banking Supervision and Regulation':
      ser2.loc[idx] = 'cole'
    elif val == 'Sandra Braunstein, Director, Division of Consumer and Community Affairs':
      ser2.loc[idx] = 'braunstein'
    elif val == 'Patrick M. Parkinson, Deputy Director, Division of Research and Statistics':
      ser2.loc[idx] = 'parkinson'
    elif val == 'Governor Susan Schmidt Bies':
      ser2.loc[idx] = 'bies'
    elif val == 'Governor Mark W. Olson':
      ser2.loc[idx] = 'olson'
    elif val == 'Governor Frederic S. Mishkin':
      ser2.loc[idx] = 'mishkin'
    elif val == 'Governor Randall S. Kroszner':
      ser2.loc[idx] = 'kroszner'
    elif val == 'Governor Kevin Warsh':
      ser2.loc[idx] = 'warsh'
    elif val == 'Sandra F. Braunstein, Director, Division of Consumer and Community Affairs ':
      ser2.loc[idx] = 'braunstein'
    elif val == 'Scott G. Alvarez, General Counsel ':
      ser2.loc[idx] = 'alvarez'
    elif val == 'David W. Wilcox, Deputy Director, Division of Research and Statistics':
      ser2.loc[idx] = 'wilcox'
    elif val == 'Patricia White, Associate Director, Division of Research and Statistics':
      ser2.loc[idx] = 'white'
    elif val == 'Vice Chairman Donald L. Kohn':
      ser2.loc[idx] = 'kohn'

  lst_s = list(ser2)
  lst_s[30] = 'brainard'
  lst_total = []
  for i in range(len(lst_d)):
    lst_total.append((i, lst_s[i], lst_d[i]))

  emp_dict = {"date" : [], "speaker" : lst_s1, "content" : []}
  for num, speaker, date in lst_total:
    emp_str = ""
    if (date == '20160414') or date == ('20110812'):
      link = 'https://www.federalreserve.gov/newsevents/testimony/' + speaker + date + '.htm'
    elif (date == '20080923'):
      link = 'https://www.federalreserve.gov/newsevents/testimony/' + speaker + date + 'a1.htm'
    else:
      link = 'https://www.federalreserve.gov/newsevents/testimony/' + speaker + date + 'a.htm'
    page = requests.get(link)
    soup = BeautifulSoup(page.text, 'html.parser')
    for i in soup.find("div", attrs = {"class":"col-xs-12 col-sm-8 col-md-8"}).find_all('p'):
      emp_str += i.get_text()
      emp_str = emp_str.replace('\xa0', ' ')
      emp_str = emp_str.replace('\n', ' ')
      emp_str = emp_str.replace('\r', ' ')
      emp_str = emp_str.replace('         ', ' ')
      emp_str = emp_str.replace('       ', ' ')
      emp_str = emp_str.replace('                ', ' ')
      emp_str = emp_str.replace('               ', ' ')
      emp_str = emp_str.replace('      ', ' ')
    emp_dict["content"].append(emp_str)
    emp_dict["date"].append(date)

  testimony0623 = pd.DataFrame(emp_dict)
  testimony0623.date = pd.to_datetime(testimony0623.date)
  testimony0623.sort_values(by='date', inplace=True)
  testimony0623.reset_index(inplace=True, drop=True)
  return testimony0623



def get_old_testimonies(): 
  links = []
  emp_dict = {'date' : [], 'speaker' : [], 'content' : []}
  years=range(1996, 1999)
  for year in years:
    page = requests.get(f'https://www.federalreserve.gov/newsevents/testimony/{year}testimony.htm')
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.select(".title")
    for i in range(len(title)):
      if (year == 1998) and (i == 3):
        links.append(title[i].find_all('a', href=True)[0]['href'])
      else:
        links.append('https://www.federalreserve.gov'+title[i].find_all('a', href=True)[0]['href'])

  for i in links:
    emp_str = ""
    txt2 = requests.get(i)
    soup = BeautifulSoup(txt2.text, 'html.parser')
    for i in soup.find_all('p'):
      emp_str += i.get_text()
      emp_str = emp_str.replace('\xa0', ' ')
      emp_str = emp_str.replace('\n', ' ')
      emp_str = emp_str.replace('\r', ' ')
      emp_str = emp_str.replace('\t', ' ')
      emp_str = emp_str.replace('         ', ' ')
      emp_str = emp_str.replace('       ', ' ')
      emp_str = emp_str.replace('                ', ' ')
      emp_str = emp_str.replace('               ', ' ')
      emp_str = emp_str.replace('      ', ' ')
      emp_str = emp_str.replace('    ', ' ')
      emp_str = emp_str.replace('  ', ' ')
    emp_dict['content'].append(emp_str)

  years1=range(1999, 2006)
  links1 = []
  for year in years1:
    page1 = requests.get(f'https://www.federalreserve.gov/newsevents/testimony/{year}testimony.htm')
    soup1 = BeautifulSoup(page1.text, 'html.parser')
    title1 = soup1.select(".title")
    for i in range(len(title1)):
      if (year == 1999) and (i == 16):
        links1.append(title1[i].find_all('a', href=True)[0]['href'])
      elif (year == 2004) and ((i == 7) or (i == 8)):
        links1.append(title1[i].find_all('a', href=True)[0]['href'])
      else:
        links1.append('https://www.federalreserve.gov'+title1[i].find_all('a', href=True)[0]['href'])

  for i in links1:
    page3 = requests.get(i)
    soup2 = BeautifulSoup(page3.text, 'html.parser')
    title2 = soup2.select("table")
    if len(str(title2[0].text))>600:
      emp_str1 = title2[0].text
    else:
      emp_str1 = title2[1].text
    emp_str1 = emp_str1.replace('\xa0', ' ')
    emp_str1 = emp_str1.replace('\n', ' ')
    emp_str1 = emp_str1.replace('\r', ' ')
    emp_str1 = emp_str1.replace('\t', ' ')
    emp_str1 = emp_str1.replace('         ', ' ')
    emp_str1 = emp_str1.replace('       ', ' ')
    emp_str1 = emp_str1.replace('                ', ' ')
    emp_str1 = emp_str1.replace('               ', ' ')
    emp_str1 = emp_str1.replace('      ', ' ')
    emp_str1 = emp_str1.replace('    ', ' ')
    emp_str1 = emp_str1.replace('  ', ' ')
    emp_dict['content'].append(emp_str1)

  years=range(1996,2006)
  for year in years:
    page2 = requests.get(f'https://www.federalreserve.gov/newsevents/testimony/{year}testimony.htm')
    soup2 = BeautifulSoup(page2.text, 'html.parser')
    title2 = soup2.select(".title")
    speakers = soup2.select(".speaker")
    for i in range(len(title2)):
      emp_dict['speaker'].append(speakers[i].text.split('\n')[1].strip())

  links_total = links + links1
  ser0 = pd.Series(links_total)
  ser0 = ser0.str.extract('(\d\d\d\d\d\d\d\d)')
  ser0 = pd.Series(ser0.values.reshape(len(ser0)))
  ser0.loc[4] = '19960718'
  ser0.loc[22] = '19970722'
  ser0.loc[31] = '19970226'
  ser0.loc[37] = '19981001'
  ser0.loc[44] = '19980721'
  ser0.loc[63] = '19980224'
  ser0.loc[69] = '19990722'
  ser0.loc[84] = '19990324'
  ser0.loc[89] = '19990223'
  ser0.loc[98] = '20000720'
  ser0.loc[111] = '20000217'
  ser0.loc[120] = '20010718'
  ser0.loc[130] = '20010228'
  ser0.loc[131] = '20010213'
  ser0.loc[137] = '20020716'
  ser0.loc[144] = '20020307'
  ser0.loc[145] = '20020227'
  ser0.loc[154] = '20030715'
  ser0.loc[161] = '20030430'
  ser0.loc[169] = '20030211'
  ser0.loc[173] = '20040720'
  ser0.loc[177] = '20040602'
  ser0.loc[178] = '20040520'
  ser0.loc[186] = '20040211'
  ser0.loc[192] = '20050720'
  ser0.loc[205] = '20050216'
  for i in ser0:
    emp_dict['date'].append(i)
  testimony9605 = pd.DataFrame(emp_dict)
  testimony9605.date = pd.to_datetime(testimony9605.date)
  testimony9605.sort_values(by='date', inplace=True)
  testimony9605.reset_index(inplace=True, drop=True)
  return testimony9605


old_testimony = get_old_testimonies()
recent_testimony = get_recent_testimonies()
total_testimony = pd.concat([old_testimony, recent_testimony], ignore_index = True)
total_testimony.to_csv("../Data/Raw Data/FOMC_Testimony.csv")