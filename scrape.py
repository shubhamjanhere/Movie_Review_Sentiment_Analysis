import openpyxl
import wikipedia
from bs4 import BeautifulSoup
import urllib3
import re
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
import json

class scrape_movie_data(object):
    """
    This function opens the excel sheet given by the user, gets the movie name and its year of release, appends it, 
    and then returns all the movies present in excel sheet in the form of a list, with '(film)' appended to it.
    """
    def search_keyword(self,filename = 'movies.xlsx'):
        work_book = openpyxl.load_workbook(filename)
        work_sheet = work_book[work_book.sheetnames[0]]
        search_keyword = []
        for cell_index in range(2,99999):
            if work_sheet['A'+str(cell_index)].value is None:
                break
            search_keyword.append(work_sheet['A'+str(cell_index)].value+' '+str(work_sheet['B'+str(cell_index)].value) +' (film)')
        return search_keyword

    """
    This function uses the movie name list from above function, searches that movie name on wikipedia and returns the 
    wikipedia page for the given movie. It then scrapes that wikipedia page, and then tries to parse the rottentomato 
    url from that wikipedia page. After scraping all the available url's, it returns it in the form of a list.
    """
    def get_rotten_home(self):
        movie_list = self.search_keyword()
        rotten_url_list = []
        for movie in movie_list:
            wiki_search = wikipedia.search(movie)
            wiki_page = wikipedia.page(wiki_search[0])  # access page url using - wiki_page.url
            http = urllib3.PoolManager()
            response = http.request('GET', wiki_page.url)
            soup = BeautifulSoup(response.data, "html.parser")
            rotten_tomato_link = list(set(re.findall(r'https://www.rottentomatoes.com/m/[\S]+/', soup.prettify())))
            if rotten_tomato_link != []:
                rotten_url_list.append(rotten_tomato_link[0])
            else:
                rotten_tomato_link = 'https://www.rottentomatoes.com/m/' + wiki_page.url.split('/')[-1]
                print("The current rotten url source is not from wikipedia, kindly debug using bellow url in case of any"
                      " error-\n" + rotten_tomato_link)
                request = requests.get(rotten_tomato_link)
                if request.status_code == 200:
                    print('The above web page exists')
                    rotten_url_list.append(rotten_tomato_link)
                else:
                    print('The above web page does not exist')
        return rotten_url_list

    """
    This function takes the homepage of any movie present in rottentomato. Using this url as argument, it creates all the 
    urls for first 30 user-review pages and then scrapes all of the user reviews. It then returns all the user reviews in
    the form of a list.
    """
    def extract_review(self, url):
        all_review = []
        for i in range(1, 31):
            if i == 1:
                review_url = url + 'reviews/?type=user'
            else:
                review_url = url + 'reviews/?page=' + str(i) + '&type=user&sort='
            http = urllib3.PoolManager()
            response = http.request('GET', review_url)
            soup = BeautifulSoup(response.data, "html.parser")
            user_review = soup.find_all("div", class_="user_review")
            if user_review:
                for review in user_review:
                    review = str(re.findall(r'</span></div>[\S\s]+</div>', str(review))).replace("['</span></div> ", "")
                    review = review.replace("</div>']", "")
                    review = review.replace('</div>"]', '')
                    review = review.replace('<br/>', ' ')
                    review = review.replace('["</span></div>', '').strip()
                    review = review.replace("[]", '')
                    all_review.append(review)
        return all_review

    """
    This function uses all above function for getting all the movie reviews for movies present in excel sheet, and then
    returns them in the form of a dictionary. It also saves all these reviews in the form of a json file for offline use.
    """
    def get_all_reviews(self):
        movie_name = self.search_keyword()
        rotten_home_url_list = self.get_rotten_home()
        all_review_dict = {}
        if len(movie_name)==len(rotten_home_url_list):
            for url in range(len(rotten_home_url_list)):
                all_review_dict[movie_name[url]] = self.extract_review(rotten_home_url_list[url])
        with open('data.json', 'w') as fp:
            json.dump(all_review_dict, fp)
        return all_review_dict

if __name__=='__main__':
    object = scrape_movie_data()
    reviews_dict = object.get_all_reviews()