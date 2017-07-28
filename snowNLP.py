from snownlp import SnowNLP
#接近1表示正面情绪，接近0表示负面情绪
text1 = '这个人脾气真坏，动不动就骂人'
text2 = '这个人脾气真好，经常笑'
s1 = SnowNLP(text1)
s2 = SnowNLP(text2)
# print(text1,s1.sentiments)
# print(text2,s2.sentiments)

text3 = '李大康就是这样的人，他穷苦出身，不攀龙附凤，不结党营私，不同流合污，不贪污受贿，也不伪造政绩，手下贪污出事了他自责用人不当，服装厂出事了也没想过隐瞒'
s = SnowNLP(text3)
# print(s.sentences)

text4 = '大师带你玩python'
s = SnowNLP(text4)
# print(s.pinyin)

s = SnowNLP(u'繁體字繁體中文的叫法在臺灣也很常見')
# print(s.han)

text5 = u'''
什么是好的演员，什么是好的演技，在人民的民义中，侯勇饰演的赵德汉小官巨贪，在别墅里人赃俱获哭得稀里哗啦片段红遍网络，这是好的演员
，这是好的演技。这就叫做爆发。
'''
s = SnowNLP(text5)
# print(s.keywords(limit=5))
# print(s.summary(limit=3))

s = SnowNLP([['性格','善良'],
             ['善良','温柔','温柔'],
             ['温柔','善良'],
             ['好人'],
             ['性格','善良']])
# print(s.tf)
# print(s.idf)

print(s.sim(['温柔']))
print(s.sim(['善良']))