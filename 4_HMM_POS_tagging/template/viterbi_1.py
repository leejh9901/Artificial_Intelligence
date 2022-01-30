"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math


def viterbi_1(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
### TRAINING ###

        tag_list = set()

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

        for seq in train:
                prev_tag = None; 
                for word, tag in seq:
                        
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


        # f = open('e.txt', 'w')
        # f.write(str(emission_counts))

        # f = open('t.txt', 'w')
        # f.write(str(transition_count))

        
        # f = open('all_tags.txt', 'w')
        # f.write(str(tag_list))
       
        # laplace smoothing constant
        e_l = 10e-7
        t_l = 10e-7


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


        for tag in emission_counts:
                total = sum(emission_counts[tag].values())
                e_len = len(emission_counts[tag])
                
                # no need to consider tag None separately bc None is already included in emission_counts
                emission_prob[tag] = {}
                for word in emission_counts[tag]:
                        # print(word)
                        # print(emission_counts[tag][word])

                        count = emission_counts[tag][word]
                        emission_prob[tag][word] = math.log((count + e_l) / (total + e_l * (e_len + 1)))
                        pass
                emission_prob[tag][None] = math.log((e_l) / (total + e_l * (e_len + 1)))
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


        f = open('e_p.txt', 'w')      
        f.write(str(emission_prob))

        f = open('t_p.txt', 'w')
        f.write(str(transition_prob))



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
      

