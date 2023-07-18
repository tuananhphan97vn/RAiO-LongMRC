import string
import unicodedata
import re
import sentencepiece as spm

# make global set punctuation
set_punctuations = set(string.punctuation)
list_punctuations_out = ['”', '”', "›", "“"]
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)


unknown_token = "<unk>"
pad_token = "<pad>"
sent_1_token = "<sent_1>"
sent_2_token = "<sent_2>"
number_token = "number"
cls_token = "[CLS]"
separate_token = "[SEP]"
mask_token = "[MASK]"
# is_next_token = "is_next"
# is_not_next_token = "is_not_next"


def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    text = re.sub(r"Ủy", "Uỷ", text)
    return text


# def norm_text_with_sub_word(text, s, convert_number=True):
#     n_arr = []
#     arr_text_sub = s.EncodeAsPieces(text)

#     for e_arr in arr_text_sub:
#         if convert_number:
#             if e_arr[0] == "▁":
#                 if not e_arr[1:].isdigit():
#                     n_arr.append(e_arr[1:])
#                 else:
#                     n_arr.append(number_token)
#             else:
#                 if not e_arr.isdigit():
#                     n_arr.append("##{}".format(e_arr))
#                 else:
#                     n_arr.append("##{}".format(number_token))
#         else:
#             if e_arr[0] == "▁":
#                 n_arr.append(e_arr[1:])
#             else:
#                 n_arr.append("##{}".format(e_arr))

#     return " ".join(n_arr).replace("\n", "")


# def check_punct_in_first_token(text):
#     # remove punct in first character like '. ' '- '

#     if text[0:2] == ". " or text[0:2] == "- ":
#         text = text[2:]
#     if text[0:3] == "-- " or text[0:3] == ".. ":
#         text = text[3:]
#     return text


def remove_multi_space(text):
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub("\s\s+", " ", text)
    # handle exception when line just all of punctuation
    if len(text) == 0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text) == 0:
        pass
    else:
        if text[-1] == " ":
            text = text[:-1]

    return "".join(text)


def replace_some_kw(text):
    text = text.replace("....", ".")
    text = text.replace("...", ".")
    text = text.replace("…", ".")
    return text


def replace_token_number(text):
    arr_text = text.split()

    l_new_arr = []
    for e_token in arr_text:
        if e_token.replace("▁", "").isdigit():
            l_new_arr.append(number_token)
        else:
            l_new_arr.append(e_token)

    return " ".join(l_new_arr)


def handle_punctuation_whole_case(text):
    # need replace | for split field in csv file

    l_new_char = []
    for e_char in text:

        if e_char not in list(set_punctuations):
            l_new_char.append(e_char)
        else:
            l_new_char.append(" {} ".format(e_char))
    text = "".join(l_new_char)

    return text


# def remove_dot_end_of_sentence(text):
#     if text[-1] == ".":
#         text = text[: -1]
#     return text


# def remove_redundant_character(text):
#     new_line_without_redundant_char = []
#     for e_char in text:
#         if e_char not in set_punctuations:
#             new_line_without_redundant_char.append(e_char)
#     return "".join(new_line_without_redundant_char)


def is_end_of_sentence(i, line):
    exception_list = [
        "Mr.",
        "MR.",
        "GS.",
        "Gs.",
        "PGS.",
        "Pgs.",
        "pgs.",
        "TS.",
        "Ts.",
        "T.",
        "ts.",
        "MRS.",
        "Mrs.",
        "mrs.",
        "Tp.",
        "tp.",
        "Kts.",
        "kts.",
        "BS.",
        "Bs.",
        "Co.",
        "Ths.",
        "MS.",
        "Ms.",
        "TT.",
        "TP.",
        "tp.",
        "ĐH.",
        "Corp.",
        "Dr.",
        "Prof.",
        "BT.",
        "Ltd.",
        "P.",
        "MISS.",
        "miss.",
        "TBT.",
        "Q.",
    ]
    if i == len(line)-1:
        return True

    if line[i+1] != " ":
        return False

    if i < len(line)-2 and line[i+2].islower():
        return False
    #
    # if re.search(r"^(\d+|[A-Za-z])\.", line[:i+1]):
    #     return False

    for w in exception_list:
        pattern = re.compile("%s$" % w)
        if pattern.search(line[:i+1]):
            return False

    return True


# # may be last line is name of author so we should remove with some condition
# def check_last_line(list_line):
#     if len(list_line[-1]) < 16:
#         list_line = list_line[:-1]
#     return list_line


def sent_tokenize(line):
    """Do sentence tokenization by using regular expression"""
    sentences = []
    cur_pos = 0
    if not re.search(r"\.", line):
        return [line]

    for match in re.finditer(r"\.", line):
        _pos = match.start()
        end_pos = match.end()
        if is_end_of_sentence(_pos, line):
            tmpsent = line[cur_pos:end_pos]
            tmpsent = tmpsent.strip()
            cur_pos = end_pos
            sentences.append(tmpsent)

    if len(sentences) == 0:
        sentences.append(line)
    elif cur_pos < len(line)-1:
        sentences.append(line[cur_pos+1:])
    return sentences


def handle_text_data_bert(text, is_uncase=True):
    if is_uncase:
        text = text.lower()
    text = normalize_text(text)
    # text = check_punct_in_first_token(text)
    text = remove_multi_space(text)
    return text


# we need option is uncase = True because can't lower text before tokenize sent
def handle_list_data_for_bert(text,
                              is_replace_tk_number=False,
                              is_uncase=True):

    list_new_text = []
   # text = handle_punctuation_whole_case(text)
    text = remove_multi_space(text)
    list_text = sent_tokenize(text)
    for e_text in list_text:
        if len(e_text) > 3:
            # e_text = remove_dot_end_of_sentence(e_text)
            e_text = handle_punctuation_whole_case(e_text)
            e_text = remove_multi_space(e_text)
            if is_replace_tk_number:
                e_text = replace_token_number(e_text)
            list_new_text.append(e_text)
    return list_new_text


if __name__ == "__main__":

    from nltk.tokenize import sent_tokenize as sentence_tok

    text = """ -LRB- Parenting.com -RRB- -- Just as her son , Mason , is walking and gabbing like a champ -- the star of TV 's `` Keeping Up With the Kardashians '' reveals the first rookie mistake that caused her to freak out and the new-mom moment that truly embarrassed her -LRB- thanks , sister Kim ! -RRB- ' I make my own baby food ' My mom bought me this amazing baby-food maker , the Beaba . -LRB- Says mom Kris Jenner : `` Who knew she would be that excited about a baby-food maker ? I bought her a million handbags , and I never got that reaction . '' -RRB- I steam and puree fruits and vegetables , and they last for like four days . Mason pretty much loves everything . I gave him red beets , and it got all over his face , which made the funniest picture . He also loves sweet potatoes , carrots , and yams . Sometimes I mix pureed peaches , pears , bananas , or apples with plain yogurt or an all-natural organic jelly . I have this great book called `` Super Baby Food '' that 's full of ideas . I should be on this Mason diet ! Parenting.com : Adorable outtakes from Kourtney and Mason 's New York photo shoot ' I have no desire to go out ' Unless I 'm working , I ca n't be away from him without feeling guilty . It does n't feel good or natural to be , so it 's a struggle . My friends keep saying , `` You and -LSB- boyfriend Scott Disick -RSB- should go out and eat or do something . '' Recently my sister Kim watched Mason so we could go to dinner . It took so long -- actually , it probably did n't take so long , but to us it felt like it took long -- we were like , should we just get pizza and go home ? Scott says he does n't need to wine and dine me anymore . We would be just as happy having a slice of pizza sitting in bed with our son . ` Mason sleeps in bed with me ' If I 've had a long day , then I have that time at night , which is really important to me . Mason did fall on the floor once by mistake . It was the worst moment . I freaked out and looked online -LSB- to research the dangers associated with a baby falling off the bed -RSB- . He was fine but crying , so I e-mailed the doctor at four in the morning . He wrote me right back , yet I stayed up all night to watch him sleep to make sure he was okay . I 've since lowered our mattress to the floor . I put pillows all around the floor , too . I am doing the best that I can , and I feel really confident in that . Unless someone has walked in your shoes , you really ca n't judge . Everyone needs to make the best choice for their life . He 's such a happy baby , and I really think sleeping together has something to do with that . Parenting.com : 15 breastfeeding celebrity moms ` I 'm still nursing ! ' I have to eat every couple of hours since I 'm nursing . If I do n't , my body freaks out . I went a little crazy for a week exercising to prepare for a photo shoot , and after that I was like , this is n't worth it . I was exhausted and dehydrated . I need to have energy for my son , and I have n't worked out since . I take Mason on a lot of walks . Carrying him around is like carrying 20 pounds all day . ` If I 'm not working , neither is the nanny ' I do n't want to judge , but I 've also met women who think it 's cool to be out or away from their baby , and I do n't get that , either . When I am out or away , that 's when I most want to be with Mason . I do have help when I 'm working . It 's important to have one person I trust , so I know Mason is taken care of . But every time I am not working , he is with me . Even on an airplane , he is with me even if the nanny is also on the plane . Any time I can have with him , I am lucky to have . I also have a big family . Out of everyone , I call -LSB- stepfather and famous Olympian Bruce Jenner -RSB- for help the most . He 's the best babysitter in the world , and a perfect role model for him . He takes Mason to the car wash for `` man time . '' Parenting.com : Celebrity moms with tattoos ` My mom is a pro -LRB- she 's had six kids ! -RRB- , but I do n't rely on her as much as people think ' I really do n't . I make my own choices . When I was pregnant , she would say , `` You need -LSB- to buy -RSB- this and this . '' I remember telling her that people used to have babies in caves . If I do n't have something , it 's not a big deal . `` She 's independent , she knows I 've had six kids , but she wants to do her own thing , '' Kris Jenner says . `` She is showing she can do it on her own . And by the way , I am impressed . '' ` We do n't really follow a set schedule ' I was struggling with that for a while , wondering what is the best thing . I think you do whatever fits your lifestyle . Some people have things planned from 7 a.m. to 7 p.m. , which works for them . But honestly , the best thing for Mason is just to be with me . When we are out , we just have so much fun . If we are cruising around , and he is napping in the car , it 's not the worst thing in the world . Parenting.com : Which baby strollers are celebs pushing ? ` Mason comes first , and if that makes me unprofessional at times , so be it ' Especially during the first three months , I did n't care if I was late . We were on his schedule . Nothing is more important than Mason . ` Being the oldest in my family , you would think I would have been in tune with kids . I had no clue ' I think motherhood is just about instinct . I remember coming home from the hospital and having no idea what we were doing . Scott and I changed his diaper together , but after a day , it was like `` Oh ! '' I got it . Even before , I had never really held babies . When my younger sisters were born , I was in high school and college . I was at my mom 's all the time but never changed them or fed them . Parenting.com : The 8 most common discipline mistakes and easy ways to avoid them ` My sister Kim is responsible for my first public -- and very embarrassing -- ` new mom ' moment ' Kim wanted to go shopping at Bergdorf Goodman , so we agreed to meet there . When we pulled up to the store , Kim had already arrived , so there were like ten paparazzi waiting . It was my first time taking a taxi with Mason . I had to tell the driver to wait because I had to get the stroller out of the car . I got Mason first , threw my bags on the ground , and was trying to get the stroller out . I 'm smiling and trying to open the stroller , but it would n't open . On top of all the paparazzi , about another 40 people are now standing there staring at me struggling . Then one lady asked if I needed help . I said , `` Yes , I would love that ! '' Everyone goes through that . I ca n't wait for the day that my sisters find out how hard it is . They try to tell me `` do this and do that , '' but they just do n't know ! Get 2 FREE YEARS of Parenting magazine - Subscribe Now !!
"""
    a = sent_tokenize(text)
    with open('sents_1.txt' ,'w') as f_w:
        for sent in a:
            f_w.write(sent + "\n")
    b =sentence_tok(text)
    print('--------------')
    with open('sents_2.txt' ,'w') as f_w:
        for sent in b:
            f_w.write(sent + "\n")

    ######check 
    print( sum([len(t.split()) for t in b]) , sum([len(t.split()) for t in a]) , len(text.split()))
