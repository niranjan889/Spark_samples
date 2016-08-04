
# coding:utf8
from pyspark import SparkContext
from pyspark import RDD
from numpy.random import RandomState

import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
to sum up:
broadcast variables and methods need to use broadcast variables need to be in the same area

unpersist files will be stored in variable broadcast broadcast variables removed immediately ,
    At a time when rdd not being triggered when rdd execution will find no broadcast variable , it will be given,
    It recommended only when the program finishes running , the broadcast variable unpersist
"""

class PLSA:

    def __init__(self, data, sc, k, is_test=False, max_itr=1000, eta=1e-6):

        """
        init the algorithm

        :type data RDD
        :param data: Enter article rdd, each record is a series of words separated by spaces , such as "I love I love the blue sky and white clouds '
        :type max_itr int
        :param max_itr: The maximum number of iterations EM
        :type is_test bool
        :param is_test: Whether the test is the rd = RandomState (1), otherwise rd = RandomState ()
        :type sc SparkContext
        :param sc: spark context
        :type k int
        :param k : Topic number
        :type eta float
        :param : Threshold , when the change log likelyhood of less than eta, stop iterating
        :return : PLSA object
        """
        self.max_itr = max_itr
        self.k = sc.broadcast(k)
        self.ori_data = data.map(lambda x: x.split(' '))
        self.sc = sc
        self.eta = eta

        self.rd = sc.broadcast(RandomState(1) if is_test else RandomState())

    def train(self):

        #Get lexicon , such as { " I " : 1}
        self.word_dict_b = self._init_dict_()
        # Words in the text , turn into the dictionary index
        self._convert_docs_to_word_index()
        # Initialization , the word distribution under each topic
        self._init_probility_word_topic_()

        pre_l= self._log_likelyhood_()

        print "L(%d)=%.5f" %(0,pre_l)

        for i in range(self.max_itr):
            #After updating each word topic posterior distribution
            self._E_step_()
            #Maximizing the lower bound
            self._M_step_()
            now_l = self._log_likelyhood_()

            improve = np.abs((pre_l-now_l)/pre_l)
            pre_l = now_l

            print "L(%d)=%.5f with %.6f%% improvement" %(i+1,now_l,improve*100)
            if improve <self.eta:
                break

    def _M_step_(self):
        """
        Update parameters p(z=k|d),p(w|z=k)
        :return: None
        """
        k = self.k
        v = self.v

        def update_probility_of_doc_topic(doc):
            """
            Updated articles relating to distribution
            """
            doc['topic'] = doc['topic'] - doc['topic']

            topic_doc = doc['topic']
            words = doc['words']
            for (word_index,word) in words.items():
                topic_doc += word['count']*word['topic_word']
            topic_doc /= np.sum(topic_doc)

            return {'words':words,'topic':topic_doc}

        self.data = self.data.map(update_probility_of_doc_topic)
        """
        rdd equivalent combination of a series of operations , and the procedure in front of the nest during subsequent operations where , when the nest more than about 60, spark will complain ,
        Where each M step through the front of the cache operation is performed off
        """
        self.data.cache()

        def update_probility_word_given_topic(doc):
            """
            Update Distribution word under each topic
            """
            probility_word_given_topic = np.matrix(np.zeros((k.value,v.value)))

            words = doc['words']
            for (word_index,word) in words.items():
                probility_word_given_topic[:,word_index] += np.matrix(word['count']*word['topic_word']).T

            return probility_word_given_topic

        probility_word_given_topic = self.data.map(update_probility_word_given_topic).sum()
        probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1))

        #So that the probability of each word and theme 1
        probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

        self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

    def _E_step_(self):
        """
        Update latent variables p (z | w, d) - Given articles, and the word , the word relating to the distribution of
        :return: None
        """
        probility_word_given_topic = self.probility_word_given_topic
        k = self.k

        def update_probility_of_word_topic_given_word(doc):
            topic_doc = doc['topic']
            words = doc['words']

            for (word_index,word) in words.items():
                topic_word = word['topic_word']
                for i in range(k.value):
                    topic_word[i] = probility_word_given_topic.value[i,word_index]*topic_doc[i]
                #So that the word and the probability distribution of each topic 1
                topic_word /= np.sum(topic_word)
            return {'words':words,'topic':topic_doc}

        self.data = self.data.map(update_probility_of_word_topic_given_word)

    def  _init_probility_word_topic_(self):
        """
        init p(w|z=k)
        :return: None
        """
        #dict length(words in dict)
        m = self.v.value

        probility_word_given_topic = self.rd.value.uniform(0,1,(self.k.value,m))
        probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1)).T

        #So that the probability of each word and theme 1
        probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

        self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

    def _convert_docs_to_word_index(self):

        word_dict_b = self.word_dict_b
        k = self.k
        rd = self.rd
        '''
        I wonder is there a better way to execute function with broadcast varible
        '''
        def _word_count_doc_(doc):
            wordcount ={}
            word_dict = word_dict_b.value
            for word in doc:
                if wordcount.has_key(word_dict[word]):
                    wordcount[word_dict[word]]['count'] += 1
                else:
                    #first one is the number of word occurance
                    #second one is p(z=k|w,d)
                    wordcount[word_dict[word]] = {'count':1,'topic_word': rd.value.uniform(0,1,k.value)}

            topics = rd.value.uniform(0, 1, k.value)
            topics = topics/np.sum(topics)
            return {'words':wordcount,'topic':topics}

        self.data = self.ori_data.map(_word_count_doc_)

    def _init_dict_(self):
        """
        init word dict of the documents,
        and broadcast it
        :return: None
        """
        words = self.ori_data.flatMap(lambda d: d).distinct().collect()
        word_dict = {w: i for w, i in zip(words, range(len(words)))}
        self.v = self.sc.broadcast(len(word_dict))
        return self.sc.broadcast(word_dict)

    def _log_likelyhood_(self):
        probility_word_given_topic = self.probility_word_given_topic
        k = self.k

        def likelyhood(doc):
            l = 0.0
            topic_doc = doc['topic']
            words = doc['words']

            for (word_index,word) in words.items():
                l += word['count']*np.log(np.matrix(topic_doc)*probility_word_given_topic.value[:,word_index])
            return l
        return self.data.map(likelyhood).sum()

    def save(self,f_word_given_topic,f_doc_topic):
        """
        Save model results TODO Add Distributed Save Results
        :param f_word_given_topic: File path for a given topic vocabulary Distribution
        :param f_doc_topic: File path for saving a document relating to the distribution of topic
        :return:
        """
        doc_topic = self.data.map(lambda x:' '.join([str(q) for q in x['topic'].tolist()])).collect()
        probility_word_given_topic = self.probility_word_given_topic.value

        word_dict = self.word_dict_b.value
        word_given_topic = []

        for w,i in word_dict.items():
            word_given_topic.append('%s %s' %(w,' '.join([str(q[0]) for q in probility_word_given_topic[:,i].tolist()])))

        f1 = open (f_word_given_topic, 'w')

        for line in word_given_topic:
            f1.write(line)
            f1.write('\n')
        f1.close()

        f2 = open (f_doc_topic, 'w')

        for line in doc_topic:
            f2.write(line)
            f2.write('\n')
        f2.close()
        print 'done'

# from PLSA import PLSA
# from pyspark import SparkContext

if __name__=="__main__":
    data = sc.textFile("092793.txt")
    
    plsa = PLSA(data,sc,3,max_itr=1)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    plsa.train()
    plsa.save('topic_word1','doc_topic1')

os.listdir(os.getcwd())

print 'h'
