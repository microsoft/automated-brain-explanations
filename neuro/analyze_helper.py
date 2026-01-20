import os
import sys
from os.path import join

import imodelsx.process_results
import numpy as np
import pandas as pd

# import viz
# import dvu
from tqdm import tqdm

sys.path.append('../experiments')
# dvu.set_style()

VOX_COUNTS = {
    'S01': 81126,
    'S02': 94251,
    'S03': 95556,
    'Mean': 90311
}
ENG1000_WORDS = sorted(set([w.lower() for w in """a, about, above, across, act, active, activity, add, afraid, after, again, age, ago, agree, air, all, alone, along, already, always, am, amount, an, and, angry, another, answer, any, anyone, anything, appear, apple, are, area, arm, army, around, arrive, art, as, ask, at, attack, aunt, autumn, away, baby, base, back, bad, bag, ball, bank, basket, bath, be, bear, beautiful, beer, bed, bedroom, behave, before, begin, behind, bell, below, besides, best, better, between, big, bird, birth, birthday, bit, bite, black, block, blood, blow, blue, board, boat, body, boil, bone, book, border, born, borrow, both, bottle, bottom, bowl, box, boy, branch, brave, bread, break, breakfast, breathe, bridge, bright, bring, brother, brown, brush, build, burn, business, bus, busy, but, buy, by, cake, call, can, candle, cap, car, card, care, careful, careless, carry, case, cat, catch, central, century, certain, chair, chance, change, chase, cheap, cheese, chicken, child, children, chocolate, choice, choose, circle, city, class, clever, clean, clear, climb, clock, cloth, clothes, cloud, cloudy, close, coffee, coat, coin, cold, collect, colour, comb, comfortable, common, compare, come, complete, computer, condition, continue, control, cook, cool, copper, corn, corner, correct, cost, contain, count, country, course, cover, crash, cross, cry, cup, cupboard, cut, dance, dangerous, dark, daughter, day, dead, decide, deep, deer, depend, desk, destroy, develop, die, different, difficult, dinner, direction, dirty, discover, dish, do, dog, door, double, down, draw, dream, dress, drink, drive, drop, dry, duck, dust, duty, each, ear, early, earn, earth, east, easy, eat, education, effect, egg, eight, either, electric, elephant, else, empty, end, enemy, enjoy, enough, enter, equal, entrance, escape, even, evening, event, ever, every, everyone, exact, everybody, examination, example, except, excited, exercise, expect, expensive, explain, extremely, eye, face, fact, fail, fall, false, family, famous, far, farm, father, fast, fat, fault, fear, feed, feel, female, fever, few, fight, fill, film, find, fine, finger, finish, fire, first, fit, five, fix, flag, flat, float, floor, flower, fly, fold, food, fool, foot, football, for, force, foreign, forest, forget, forgive, fork, form, fox, four, free, freedom, freeze, fresh, friend, friendly, from, front, fruit, full, fun, funny, furniture, further, future, game, garden, gate, general, gentleman, get, gift, give, glad, glass, go, goat, god, gold, good, goodbye, grandfather, grandmother, grass, grave, great, green, grey, ground, group, grow, gun, hair, half, hall, hammer, hand, happen, happy, hard, hat, hate, have, he, head, healthy, hear, heavy, hello, help, heart, heaven, height, help, her, here, hers, hide, high, hill, him, his, hit, hobby, hold, hole, holiday, home, hope, horse, hospital, hot, hotel, house, how, hundred, hungry, hour, hurry, husband, hurt, I, ice, idea, if, important, in, increase, inside, into, introduce, iron, invite, is, island, it, its, jelly, job, join, juice, jump, just, keep, key, kill, kind, king, kitchen, knee, knife, knock, know, ladder, lady, lamp, land, large, last, late, lately, laugh, lazy, lead, leaf, learn, leave, leg, left, lend, length, less, lesson, let, letter, library, lie, life, light, like, lion, lip, list, listen, little, live, lock, lonely, long, look, lose, lot, love, low, lower, luck, machine, main, make, male, man, many, map, mark, market, marry, matter, may, me, meal, mean, measure, meat, medicine, meet, member, mention, method, middle, milk, million, mind, minute, miss, mistake, mix, model, modern, moment, money, monkey, month, moon, more, morning, most, mother, mountain, mouth, move, much, music, must, my, name, narrow, nation, nature, near, nearly, neck, need, needle, neighbour, neither, net, never, new, news, newspaper, next, nice, night, nine, no, noble, noise, none, nor, north, nose, not, nothing, notice, now, number, obey, object, ocean, of, off, offer, office, often, oil, old, on, one, only, open, opposite, or, orange, order, other, our, out, outside, over, own, page, pain, paint, pair, pan, paper, parent, park, part, partner, party, pass, past, path, pay, peace, pen, pencil, people, per, perfect, period, person, photograph, piano, pick, picture, piece, pig, pin, pink, place, plane, plant, plastic, plate, play, please, pleased, plenty, pocket, point, poison, police, polite, pool, poor, popular, position, possible, potato, pour, power, present, press, pretty, prevent, price, prince, prison, private, prize, probably, problem, produce, promise, proper, protect, provide, public, pull, punish, pupil, push, put, queen, question, quick, quiet, quite, radio, rain, raise, reach, read, ready, real, really, receive, record, red, remember, remind, remove, rent, repair, repeat, reply, report, rest, restaurant, result, return, rice, rich, ride, right, ring, rise, road, rob, rock, room, round, rubber, rude, rule, ruler, run, rush, sad, safe, sail, salt, same, sand, save, say, school, science, search, seat, second, see, seem, sell, send, sentence, serve, seven, several, sex, shade, shadow, shake, shape, share, sharp, she, sheep, sheet, shelf, shine, ship, shirt, shoe, shoot, shop, short, should, shoulder, shout, show, sick, side, signal, silence, silly, silver, similar, simple, single, since, sing, sink, sister, sit, six, size, skill, skin, skirt, sky, sleep, slip, slow, smoke, small, smell, smile, smoke, snow, so, soap, sock, soft, some, someone, something, sometimes, son, soon, sorry, sound, soup, south, space, speak, special, speed, spell, spend, spoon, sport, spread, spring, square, stamp, stand, star, start, station, stay, steal, steam, step, still, stomach, stone, stop, store, storm, story, strange, street, strong, structure, student, study, stupid, subject, substance, successful, such, sudden, sugar, suitable, summer, sun, sunny, support, sure, surprise, sweet, swim, sword, table, take, talk, tall, taste, taxi, tea, teach, team, tear, telephone, television, tell, ten, tennis, terrible, test, than, that, the, their, then, there, therefore, these, thick, thin, thing, think, third, this, though, threat, three, tidy, tie, title, to, today, toe, together, tomorrow, tonight, too, tool, tooth, top, total, touch, town, train, travel, tree, trouble, true, trust, two, twice, try, turn, type, uncle, under, understand, unit, until, up, use, useful, usual, usually, vegetable, very, village, voice, visit, wait, wake, walk, want, warm, wash, waste, watch, water, way, we, weak, wear, weather, wedding, week, weight, welcome, well, west, wet, what, wheel, when, where, which, while, white, who, why, wide, wife, wild, will, win, wind, window, wine, winter, wire, wise, wish, with, without, woman, wonder, word, work, world, worry, worst, write, wrong, year, yes, yesterday, yet, you, young, your, zero""".split(", ")]))


def abbrev_question(q):
    q = q.replace('Is time mentioned in the input?',
                  'Does the input mention time?')
    q = q.replace('express a sense of belonging or connection to a place or community',
                  'express a connection to a community')
    q = q.replace('express the narrator\'s opinion or judgment about an event or character',
                  'express an opinion about an event or character')
    q = q.replace('involve a description of', 'describe a')
    q = q.replace('involve a discussion about', 'discuss')
    q = q.replace('involve an expression of', 'express')
    q = q.replace('involve the mention of', 'mention')
    q = q.replace('that leads to a change or revelation', '')
    q = q.replace(' ?', '?')
    for prefix in ['Does the sentence ', 'Is the sentence ', 'Does the input ', 'Is the input ', 'Does the text ', 'Is there a ', 'Is an ', 'Is a ', 'Is there ',]:
        q = q.replace(prefix, '...')
    return q


def abbrev_questions(qs):
    return [abbrev_question(q) for q in qs]


def load_results(save_dir):
    dfs = []
    fnames = [
        fname for fname in os.listdir(save_dir)[::-1]
        if not fname.startswith('coef')
    ]
    for fname in tqdm(fnames):
        df = pd.read_pickle(join(save_dir, fname))
        # print(fname)
        # display(df)
        dfs.append(df.reset_index())
    d = pd.concat(dfs)
    # d = d.drop(columns='coef_')
    # .round(2)
    # d.set_index(['feats', 'dset'], inplace=True)
    d['nonlin_suffix'] = d['nonlinearity'].fillna(
        '').str.replace('None', '').str.replace('tanh', '_tanh')
    d['model'] = d['model'] + d['nonlin_suffix']
    d['model_full'] = d['model'] + '_thresh=' + \
        d['perc_threshold_fmri'].astype(str)
    return d


def load_clean_results(results_dir, experiment_filename='../experiments/02_fit_encoding.py'):
    # load the results in to a pandas dataframe
    r = imodelsx.process_results.get_results_df(results_dir)
    r = imodelsx.process_results.fill_missing_args_with_default(
        r, experiment_filename)
        
    for k in ['save_dir', 'save_dir_unique']:
        r[k] = r[k].map(lambda x: x if x.startswith('/home')
                        else x.replace('/mntv1', '/home/chansingh/mntv1'))
    if 'use_huge' in r.columns:
        r = r.drop(columns=['use_huge'])
    r['qa_embedding_model'] = r.apply(lambda row: {
        'mistralai/Mistral-7B-Instruct-v0.2': 'mist-7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mixt-moe',
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama3-8B',
        'meta-llama/Meta-Llama-3-8B-Instruct-fewshot': 'llama3-8B-fewshot',
        'meta-llama/Meta-Llama-3-8B-Instruct-refined': 'llama3-8B-refined',
    }.get(row['qa_embedding_model'], row['qa_embedding_model']) if 'qa_emb' in row['feature_space'] else '', axis=1)
    r['subject'] = r['subject'].str.replace('UTS', 'S')
    r['qa_questions_version'] = r.apply(
        lambda row: row['qa_questions_version'] if 'qa_emb' in row['feature_space'] else '', axis=1)
    mets = [c for c in r.columns if 'corrs' in c and (
        'mean' in c or 'frac' in c)]
    cols_varied = imodelsx.process_results.get_experiment_keys(
        r, experiment_filename)
    print('experiment varied these params:', cols_varied)
    r['corrs_test_mean_sem'] = r['corrs_test'].apply(
        lambda x: np.std(x) / np.sqrt(len(x)))
    r['feature_space_simplified'] = r['feature_space'].apply(
        lambda x: 'llama' if 'llama' in x else x
    )
    mets.append('corrs_test_mean_sem')
    return r, cols_varied, mets
