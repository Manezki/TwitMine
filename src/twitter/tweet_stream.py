#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
import tweepy
import webbrowser
import json

#Variables that contains the user credentials to access Twitter API 
access_token = "3437775623-94crhJjZxeYPHjs9QXecpWcYPF98hNJSyqnSjfM"
access_token_secret = "ASQsqVB68B1Ai57EyHiVOQZmY8sibXiBmgDUkP9St9dOu"
consumer_key = "vlZJET4SaRsgaFZdYvRuAEQlo"
consumer_secret = "uLDOghXZXynRWf9IdwPZHyx32hNG78VbuQBCbvErrvpLdRyq8j"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        data = json.loads(data)
        print(data['text'])
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)


    stream = tweepy.Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    #stream.filter(track=['bitcoin', 'ethereum', 'bitcoin cash', 'ripple', 'litecoin', 'cardano', 'iota', 'dash', 'nem', 'bitcoin gold', 'monero', 'stellar'], languages=["en"])
    stream.filter(languages=["en"], track=["Santa Claus Village", "Arctic Circle", "Lapland", "Finland"])