from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re


#Convert abbreviation to full text
def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   }
   
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (abbrev)

#Remove unnecessary text
def remove_features(data_str):
   
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')    
    mention_re = re.compile(r'@|#(\w+)')  
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)  
    data_str = data_str.lower()  
    data_str = url_re.sub(' ', data_str)   
    data_str = mention_re.sub(' ', data_str)  
    data_str = num_re.sub(' ', data_str)
    return data_str

   
#Set polarity range from TextBlob library
def polarity(value):
    
    if value > 0:
        return "+ve"
    elif value < 0:
        return "-ve"
    else:
        return "neu"

   
#Main function
def main(sc, filename):
    
    data = sc.textFile(filename)\
           .map(lambda x:x.split(','))\
           .filter(lambda x:len(x)==8)\
           .filter(lambda x:len(x[0])>1)\
           .map(lambda x:x[4])\
           .map(lambda x:abb_en(x))\
           .map(lambda x:remove_features(x))\
           .map(lambda x:TextBlob(x).sentiment.polarity)\
           .map(lambda x:polarity(x))
            
    data_raw = sc.textFile(filename)\
               .map(lambda x:x.split(','))\
               .filter(lambda x:len(x)==8)\
               .filter(lambda x:len(x[0])>1)\
               .map(lambda x:(x[4],x[0],x[2],x[1],x[3],x[5],x[6],x[7]))
      
    
    data_zip = data.zip(data_raw).map(lambda x:str(x).replace("'","").replace('"',""))
    data_zip.saveAsTextFile("DE22C04_Afif")
   
#Specify no. of threads and AppName
if __name__ == "__main__":
    
    conf = SparkConf().setMaster("local[1]").setAppName("DE22C04_Test_Afif")
    sc = SparkContext (conf=conf)
    
    filename = "starbuck_v1.csv"
  
    main(sc, filename)

    sc.stop()
