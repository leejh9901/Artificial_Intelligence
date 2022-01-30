"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math

# def LaplaceSmoothingHapax(word, tag):
        

def viterbi_3(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        

        tag_list = set()
        hapax_list = dict()     # key: word, value: 1 if hapax, 0 if not hapax
        
        suffix_tatal_sum = 0
        suffix_list = {"ed", "ness", "able", "ward", "less", "ment", "wise", "ious", "ship", "ible", "tion", "ate", "ful", "ly", "ize"}
        # suffix_list = {"ness", "able", "ward", "less", "ment", "wise", "ious", "ship", "ible", "tion"}

        suffix_counts = dict()  # key: suffix; value: {tag:count}
        suffix_counts = {"ed":{}, "ness":{}, "able":{}, "ward":{}, "less":{}, "ment":{}, "wise":{}, "ious":{}, "ship":{}, "ible":{}, "tion":{}, "ate":{}, "ful":{}, "ly":{}, "ize":{}}
        # suffix_counts = {"ness":{}, "able":{}, "ward":{}, "less":{}, "ment":{}, "wise":{}, "ious":{}, "ship":{}, "ible":{}, "tion":{}}

        '''
                {
                        suffix: {
                                tag1: count1,
                                tag2: count2, 
                                ...
                        }

                }

                suffix1: ness   NOUN
                suffix2: ly     ADV
                suffix3: able   ADJ
                suffix4: ward   ADV
                suffix5: ize    VERB
                suffix6: less   ADJ
                suffix7: ment   NOUN
                suffix8: ate    VERB
                suffix9: ful    ADJ
                suffix10: wise  ADV
                suffix11: ious  ADJ
                suffix12: ship  NOUN
                suffix13: ible  ADJ
                suffix14: tion  NOUN

                # NOUN
                # if word[-1] == 's' and word[-2] == 's' and word[-3] == 'e' and word[-4] == 'n' and tag == "NOUN":
                if (word[-3:-1] == "acy" or word[-4:-1] == "ance" or word[-4:-1] == "ence" or word[-3:-1] == "dom") and tag == "NOUN" :
                        c += 500
                elif (word[-2:-1] == "er" or word[-2:-1] == "or" or word[-3:-1] == "ist" or word[-3:-1] == "ian" or word[-3:-1] == "eer") and tag == "NOUN":
                        c += 500
                elif (word[-2:-1] == "ty" or word[-3:-1] == "ity" or word[-4:-1] == "ment" or word[-4:-1] == "ness" or word[-4:-1] == "ship" or word[-4:-1] == "sion" or word[-4:-1] == "tion") and tag == "NOUN":
                        c += 500
                # ADV
                # elif word[-1] == 'y' and word[-2] == 'l' and tag == "ADV":
                elif word[-2:-1] == "ly" and tag == "ADV" :
                        c += 500
                elif (word[-4:-1] == "ward" or word[-5:-1] == "wards" or word[-4:-1] == "wise") and tag == "ADV":
                        c += 500
                # elif len(word) > 4 and word[-4:-1] == "wise" and tag == "ADV":
                #         c += 100
                # ADJ
                elif (word[-4:-1] == "able" or word[-4:-1] == "ible" or word[-3:-1] == "ful" or word[-2:-1] == 'al' or word[-2:-1] == "ic" or word[-4:-1] == "ical") and tag == "ADJ":
                        c += 500
                elif (word[-4:-1] == "ious" or word[-3:-1] == "ous" or word[-3:-1] == "ish" or word[-3:-1] == "ive" or word[-4:-1] == "less") and tag == "ADJ":
                        c += 500
                # VERB
                elif (word[-3:-1] == "ate" or word[-2:-1] == "en" or word[-3:-1] == "ify" or word[-2:-1] == "fy" or word[-3:-1] == "ize" or word[-3:-1] == "ing" or word[-2:-1] == "ed") and tag == "VERB" :
                        c += 500
                # elif word[-1] == 'd' and word[-2] == 'e' and tag == "VERB":
                # elif word[-2:-1] == "ed" and tag == "VERB" :
                #         c += 100
                elif 48 <= ord(word[0]) and ord(word[0]) <= 57:
                        for i in range(1,len(word)):
                                if not (48 <= ord(word[i]) and ord(word[i]) <= 57):
                                        break
                                if i == (len(word)-1) :
                                        c += 100000
                
                                
                            
        '''

        emission_counts = dict() 
        '''
                {
                        tag: [total, {
                                word1: count1
                        }]
                }

                P(w1| t1) = count1 / total
        '''

        transition_count = dict()
        '''
                {
                        None: [12312, {
                                ...
                        }],
                        prev_tag: [total_count,  {
                                next_tag_1: count1,
                                next_tag2: count2
                        }], 
                        prev_tag2: [total_count,  {
                                next_tag_1: count1,
                                next_tag2: count2
                        }], 
                }

                P(t1|pt) = count1 / (count1+count2) = count1 / total
        '''
        
        # seq = train[32]
        # if (1):
        # for seq in [train[29], train[211]]:
        for seq in train:
                prev_tag = None; 
                for word, tag in seq:
                        # SUFFIX
                        if (len(word) > 3) :
                                # if tag not in suffix_counts:
                                #         suffix_counts[tag] = {}

                                for suf in suffix_list:
                                        if suf not in suffix_counts:
                                                suffix_counts[suf] = {}
                                        if word[-4:-1] == suf :
                                                suffix_tatal_sum += 1
                                                if tag not in suffix_counts[suf]:
                                                        suffix_counts[suf][tag] = 1
                                                else:
                                                        suffix_counts[suf][tag] += 1

                        # # if (len(word) > 3) :
                        #         # if tag not in suffix_counts:
                        #         #         suffix_counts[tag] = {}
                        # for suf in suffix_list:
                        #         if suf not in suffix_counts:
                        #                 suffix_counts[suf] = {}
                        #         # print([word, suf])
                        #         # if suf == "ed":
                        #         #         if word[-2:] == suf:
                        #         #                 suffix_tatal_sum += 1
                        #         #                 if tag not in suffix_counts[suf]:
                        #         #                         suffix_counts[suf][tag] = 1
                        #         #                 else:
                        #         #                         suffix_counts[suf][tag] += 1
                        #         if word[-len(suf):] == suf :
                        #                 # print("REACHED")
                        #                 suffix_tatal_sum += 1
                        #                 if tag not in suffix_counts[suf]:
                        #                         suffix_counts[suf][tag] = 1
                        #                 else:
                        #                         suffix_counts[suf][tag] += 1


                        if word in hapax_list:  # not hapax
                                hapax_list[word] = 0
                                # for suf in suffix_list:
                                #         # if suf not in suffix_counts:
                                #         #         suffix_counts[suf] = {}
                                #         # print([word, suf])
                                #         # if suf == "ed":
                                #         #         if word[-2:] == suf:
                                #         #                 suffix_tatal_sum -= 1
                                #         #                 suffix_counts[suf][tag] -= 1
                                #         #                 # if tag not in suffix_counts[suf]:
                                #         #                 #         suffix_counts[suf][tag] = 1
                                #         #                 # else:
                                #         #                 #         suffix_counts[suf][tag] += 1
                                #         if word[-len(suf):] == suf :
                                #                 # print("REACHED")
                                #                 suffix_tatal_sum -= 1
                                #                 suffix_counts[suf][tag] -= 1
                                #                 # if tag not in suffix_counts[suf]:
                                #                 #         suffix_counts[suf][tag] = 1
                                #                 # else:
                                #                 #         suffix_counts[suf][tag] += 1

                        else :
                                hapax_list[word] = 1
                                # for suf in suffix_list:
                                #         # if suf == "ed":
                                #         #         if word[-2:] == suf:
                                #         #                 suffix_tatal_sum += 1
                                #         #                 if tag not in suffix_counts[suf]:
                                #         #                         suffix_counts[suf][tag] = 1
                                #         #                 else:
                                #         #                         suffix_counts[suf][tag] += 1
                                #         if word[-len(suf):] == suf :
                                #                 suffix_tatal_sum += 1
                                #                 if tag not in suffix_counts[suf]:
                                #                         suffix_counts[suf][tag] = 1
                                #                 else:
                                #                         suffix_counts[suf][tag] += 1


                       # if tag is an unseen tag
                        if tag not in emission_counts:
                                emission_counts[tag] = {word : 1}
                        else: # if tag is already seen
                                # if word is an unseen word with a seen tag
                                if word not in emission_counts[tag]:
                                        emission_counts[tag][word] = 1
                                else: # if word is already seen with a seen tag
                                        emission_counts[tag][word] += 1

                        
                        #
                        if prev_tag not in transition_count:
                                transition_count[prev_tag] = {tag : 1}
                        else:
                                if tag not in transition_count[prev_tag]:
                                        transition_count[prev_tag][tag] = 1
                                else:
                                        transition_count[prev_tag][tag] += 1

                        tag_list.add(tag)
                        prev_tag = tag
                        pass; 
                pass; 


        hapax_words = [key for (key, value) in hapax_list.items() if value == 1]
        # temp_counts = [value for (key, value) in emission_counts.items() if key in hapax_words]
        hapax_count_tag = dict()  
        '''
                { all the words are hapaxwords
                        tag1: {word1: count1},
                        tag2: {word2: count2},
                        ...
                }
        '''
        hapax_total_sum = 0
        for tag in emission_counts:
                hapax_count_tag[tag] = dict()
                for word in emission_counts[tag]:
                        if word in hapax_words:
                                hapax_count_tag[tag][word] = emission_counts[tag][word]
                                hapax_total_sum += emission_counts[tag][word]


        # print(hapax_words)
        # print(hapax_count_tag)
        # b = [value for (key, value) in hapax_list.items() if key in a]
        # print(hapax_words)
        # print(b)
        # print(len(a))
        # print(len(b))
        # print(len(hapax_list))


        f = open('s.txt', 'w')
        f.write(str(suffix_counts))

        # f = open('t.txt', 'w')
        # f.write(str(transition_count))

        
        # f = open('all_tags.txt', 'w')
        # f.write(str(tag_list))
  ###################     
        # laplace smoothing constant
        e_l = 10e-10
        t_l = 10e-10

        # e_l = 10e-5
        # t_l = 10e-5


        emission_prob = dict()
        '''
                {
                        tag1: {
                                word1: P(word1| tag1)
                        }

                }
        '''
        
        transition_prob = dict()

        '''
                {
                        prev: {
                                tag: ln(P(tag | prev)) = ln( (count + lam) / (total + lam*(V+1)) ),
                                ...
                                None: ln( (lam) / (total + V(landwe)) )
                        }
                }
        '''
        # new_word_tag = []
        # new_word_tag = 
        


        for tag in emission_counts:
                total = sum(emission_counts[tag].values())
                e_len = len(emission_counts[tag])
                hapax_total = sum(hapax_count_tag[tag].values())
                # suffix_total = sum(suffix_counts[tag].values())
                
                # no need to consider tag None separately bc None is already included in emission_counts
                emission_prob[tag] = {}
                for word in emission_counts[tag]:
                        # print(word)
                        # print(emission_counts[tag][word])

                        count = emission_counts[tag][word]
                        mult_e_l = 1
                        if word in hapax_words:
                                c = 1 # * hapax_total / hapax_total_sum 
                                # SUFFIX
                                if (len(word) > 3) :
                                        for suf in suffix_list:
                                                if suf == "ed":
                                                        if word[-2:] == suf:
                                                                c =  100 * suffix_counts[suf][tag] / sum(suffix_counts[suf].values())

                                                elif word[-len(suf):] == suf :
                                                        # print("REACHED")
                                                        if tag in suffix_counts[suf]:
                                                                c = 100 * suffix_counts[suf][tag] / sum(suffix_counts[suf].values())
                                                        else:
                                                                c = 100 * 10e-10

                                            
                                mult_e_l =  1000* c * hapax_total / hapax_total_sum # hapax_count_tag[tag][word]
                                # if word[]
                        emission_prob[tag][word] = math.log((count + e_l * mult_e_l) / (total + e_l * mult_e_l * (e_len + 1)))
                        pass
                emission_prob[tag][None] = math.log((e_l * mult_e_l) / (total + e_l * mult_e_l * (e_len + 1)))
                pass


        for prev in transition_count:
                total = sum(transition_count[prev].values())
                t_len = len(transition_count[prev])

                transition_prob[prev] = {}
                transition_prob[prev][None] = math.log(t_l / (total + t_l * (t_len + 1)))
                for tag in transition_count[prev]:
                        count = transition_count[prev][tag]
                        transition_prob[prev][tag] = math.log((count + t_l) / (total + t_l * (t_len + 1)))
                        pass

                pass


        # f = open('e_p.txt', 'w')      
        # f.write(str(emission_prob))

        # f = open('t_p.txt', 'w')
        # f.write(str(transition_prob))



        # print(test[60])
        '''
                trellis = [
                        [probability, current_tag, path]
                ]
        '''

        # seq = test[348]
        final_path = []

        for seq in test:
                trellis = []

                for tag in tag_list:
                        trellis.append([0, None, []])
                        pass

                for word in seq:
                        next_trellis = []
                        for tag in tag_list:
                                best_trellis = None
                                for prev_prob, prev_tag, prev_path in trellis:
                                        word_emission = emission_prob[tag].get(word, emission_prob[tag][None])
                                        
                                        prev_tag_transition_probs = transition_prob.get(prev_tag, transition_prob[None])
                                        tag_transition = prev_tag_transition_probs.get(tag, prev_tag_transition_probs[None])

                                        curr_prob = prev_prob + word_emission + tag_transition
                                        if best_trellis == None or curr_prob > best_trellis[0]:
                                                path = prev_path[:]
                                                path.append( (word, tag) )
                                                best_trellis = [curr_prob, tag, path]

                        
                                next_trellis.append(best_trellis)
                        trellis = next_trellis


                # print(trellis)

                best_prob = -math.inf

                for prev_prob, prev_tag, prev_path in trellis:
                        if prev_prob > best_prob:
                                best_prob = prev_prob
                                best_path = prev_path
                
                # f = open('pred.txt', 'w')
                # f.write(str(trellis))


                # print(best_path)
                final_path.append(best_path)
        
        return final_path



        return []