# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 18:13:31 2021

@author: Asus
"""
import numpy


def read():
    # with open("./"+inp_file, encoding='utf8') as fin:
    #     lines = fin.readlines()
    #     print(len(lines))
    fin_train = open('train_set.txt', 'r')
    lines_train = fin_train.readlines()
    fin_train.close()

    fin_test = open('test_set.txt', 'r')
    lines_test = fin_test.readlines()
    fin_test.close()

    fin_dev = open('dev_set.txt', 'r')
    lines_dev = fin_dev.readlines()
    fin_dev.close()
    # print(lines[1])

    y_train = []
    y_test = []
    y_dev = []

    xchr_train = []
    xchr_test = []
    xchr_dev = []

    xwrd_train = []
    xwrd_test = []
    xwrd_dev = []

    kchr = 3
    kwrd = 5
    wordcnt = 0
    charcnt = 0
    wordsmap = {}
    charmap = {}

    maxwordlen, maxsenlen = 0, 0
    # print(len(lines))
    # numsent = len(lines)
    # numsent = 21000
    # numsent = 50
    numsent_train = len(lines_train)
    numsent_test = len(lines_test)
    numsent_dev = len(lines_dev)

    # print(numsent)
    # print(len(lines))

    for line in lines_train[0:numsent_train]:
        words = line.split()
        tokens = words[:-1]
        # print(tokens)
        y_train.append(int(float(words[-1])))
        maxsenlen = max(maxsenlen, len(tokens))
        for token in tokens:
            if token not in wordsmap:
                wordsmap[token] = wordcnt
                wordcnt += 1
                maxwordlen = max(maxwordlen, len(token))
            for i in range(len(token)):
                if token[i] not in charmap:
                    charmap[token[i]] = charcnt
                    charcnt += 1
        # print(wordsmap)
        # print(charmap)
        # break
    # print(y)

    for line in lines_test[0:numsent_test]:
        words = line.split()
        tokens = words[:-1]
        y_test.append(int(float(words[-1])))
        maxsenlen = max(maxsenlen, len(tokens))
        for token in tokens:
            if token not in wordsmap:
                wordsmap[token] = wordcnt
                wordcnt += 1
                maxwordlen = max(maxwordlen, len(token))
            for i in range(len(token)):
                if token[i] not in charmap:
                    charmap[token[i]] = charcnt
                    charcnt += 1
        # print(wordsmap)
        # print(charmap)
        # break
    # # print(y)
    #
    #
    for line in lines_dev[0:numsent_dev]:
        words = line.split()
        tokens = words[:-1]
        y_dev.append(int(float(words[-1])))
        maxsenlen = max(maxsenlen, len(tokens))
        # print(maxsenlen)
        for token in tokens:
            if token not in wordsmap:
                wordsmap[token] = wordcnt
                wordcnt += 1
                maxwordlen = max(maxwordlen, len(token))
            for i in range(len(token)):
                if token[i] not in charmap:
                    charmap[token[i]] = charcnt
                    charcnt += 1
        # print(wordsmap)
        # print(charmap)
        # break
    # print(y)
    #

    for line in lines_train[0:numsent_train]:
        words = line.split()
        tokens = words[:-1]
        # print(tokens)
        wordmat = [0] * (maxsenlen + kwrd - 1)
        # print(wordmat)
        charmat = numpy.zeros((maxsenlen + kwrd - 1, maxwordlen + kchr - 1))
        # print(charmat)
        for i in range(len(tokens)):
            # print(i)
            # print(tokens[i])
            # print(int((kwrd / 2) + i))
            # print(wordsmap[tokens[i]])
            # print("**********")
            wordmat[int((kwrd / 2) + i)] = wordsmap[tokens[i]]
            for j in range(len(tokens[i])):
                charmat[int((kwrd / 2) + i)][int((kchr / 2) + j)] = charmap[tokens[i][j]]

        xchr_train.append(charmat)
        xwrd_train.append(wordmat)

    for line in lines_test[0:numsent_test]:
        words = line.split()
        tokens = words[:-1]
        # print(tokens)
        wordmat = [0] * (maxsenlen + kwrd - 1)
        # print(wordmat)
        charmat = numpy.zeros((maxsenlen + kwrd - 1, maxwordlen + kchr - 1))
        # print(charmat)
        for i in range(len(tokens)):
            # print(i)
            # print(tokens[i])
            # print(int((kwrd / 2) + i))
            # print(wordsmap[tokens[i]])
            # print("**********")
            wordmat[int((kwrd / 2) + i)] = wordsmap[tokens[i]]
            for j in range(len(tokens[i])):
                charmat[int((kwrd / 2) + i)][int((kchr / 2) + j)] = charmap[tokens[i][j]]

        xchr_test.append(charmat)
        xwrd_test.append(wordmat)
    # print(maxsenlen)
    # print(maxwordlen)


    for line in lines_dev[0:numsent_dev]:
        words = line.split()
        tokens = words[:-1]
        # print(tokens)
        wordmat = [0] * (maxsenlen + kwrd - 1)
        # print(wordmat)
        charmat = numpy.zeros((maxsenlen + kwrd - 1, maxwordlen + kchr - 1))
        # print(charmat)
        for i in range(len(tokens)):
            # print(i)
            # print(tokens[i])
            # print(int((kwrd / 2) + i))
            # print(wordsmap[tokens[i]])
            # print("**********")
            wordmat[int((kwrd / 2) + i)] = wordsmap[tokens[i]]
            for j in range(len(tokens[i])):
                charmat[int((kwrd / 2) + i)][int((kchr / 2) + j)] = charmap[tokens[i][j]]

        xchr_dev.append(charmat)
        xwrd_dev.append(wordmat)

    #
    maxwordlen += kchr - 1
    maxsenlen += kwrd - 1
    #
    # # print(xchr)
    # # print(xwrd)
    # # break
    #
    # #         # print(len(wordmat))
    # #         # print(tokens)

    data = (charmap, wordsmap, numsent_train, numsent_test, numsent_dev, charcnt, wordcnt, maxwordlen, maxsenlen, kchr, kwrd, xchr_train, xchr_test, xchr_dev, xwrd_train, xwrd_test, xwrd_dev, y_train, y_test, y_dev)
    # # print(wordcnt)
    return data


if __name__ == '__main__':
    # read("tweets_clean.txt")
    read()



