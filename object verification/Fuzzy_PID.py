# coding: utf-8
import numpy as np
import torch
import skfuzzy as fuzz

def fuzzyP3(x, max_err):
    membership = [0, 0, 0]
    bisection_error = 1/3 * max_err
    if x <= 1*bisection_error:
        membership[0] = 1
    elif 1*bisection_error < x and x <= 2*bisection_error:
        membership[0] = (2*bisection_error - x) / bisection_error

    if 1*bisection_error < x and x <= 2*bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2*bisection_error < x and x <= 3*bisection_error:
        membership[1] = (3*bisection_error - x) / bisection_error

    if 2*bisection_error < x and x <= 3*bisection_error:
        membership[2] = (x - 2*bisection_error) / bisection_error
    elif 3*bisection_error <= x:
        membership[2] = 1
    return membership

def fuzzyP5(x, max_err):
    membership = [0, 0, 0, 0, 0]
    bisection_error = 1/5 * max_err
    if x <= 1*bisection_error:
        membership[0] = 1
    elif 1*bisection_error < x and x <= 2*bisection_error:
        membership[0] = (2*bisection_error - x) / bisection_error

    if 1*bisection_error < x and x <= 2*bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2*bisection_error < x and x <= 3*bisection_error:
        membership[1] = (3*bisection_error - x) / bisection_error

    if 2*bisection_error < x and x <= 3*bisection_error:
        membership[2] = (x - 2*bisection_error) / bisection_error
    elif 3*bisection_error < x and x <= 4*bisection_error:
        membership[2] = (4*bisection_error - x) / bisection_error

    if 3*bisection_error < x and x <= 4*bisection_error:
        membership[3] = (x - 3*bisection_error) / bisection_error
    elif 4*bisection_error < x and x <= 5*bisection_error:
        membership[3] = (5*bisection_error - x) / bisection_error

    if 4*bisection_error < x and x <= 5*bisection_error:
        membership[4] = (x - 4*bisection_error) / bisection_error
    elif 5*bisection_error <= x:
        membership[4] = 1
    return membership

def fuzzyP7(x, max_err):
    # input must be rescaled
    # output is a list of membership
    # [NB, NM, NS, ZE, PS, PM, PB]
    membership = [0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1/7 * max_err
    # NB
    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2*bisection_error:
        membership[0] = (2*bisection_error - x) / bisection_error
    # NM
    if bisection_error < x and x <= 2*bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2*bisection_error < x and x <= 3*bisection_error:
        membership[1] = (3*bisection_error - x) / bisection_error
    # NS
    if 2*bisection_error < x and x <= 3*bisection_error:
        membership[2] = (x - 2*bisection_error) / bisection_error
    elif 3*bisection_error < x and x <= 4*bisection_error:
        membership[2] = (4*bisection_error - x) / bisection_error
    # ZE
    if 3*bisection_error < x and x <= 4*bisection_error:
        membership[3] = (x - 3*bisection_error) / bisection_error
    elif 4*bisection_error < x and x <= 5*bisection_error:
        membership[3] = (5*bisection_error - x) / bisection_error
    # PS
    if 4*bisection_error < x and x <= 5*bisection_error:
        membership[4] = (x - 4*bisection_error) / bisection_error
    elif 5*bisection_error < x and x <= 6*bisection_error:
        membership[4] = (6*bisection_error - x) / bisection_error
    # PM
    if 5*bisection_error < x and x <= 6*bisection_error:
        membership[5] = (x - 5*bisection_error) / bisection_error
    elif 6*bisection_error < x and x <= 7*bisection_error:
        membership[5] = (7*bisection_error - x) / bisection_error
    # PB
    if 6*bisection_error < x and x <= 7*bisection_error:
        membership[6] = (x - 6*bisection_error) / bisection_error
    elif 7*bisection_error <= x:
        membership[6] = 1
    return membership


def fuzzyP9(x, max_err):
    membership = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1/9 * max_err

    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2*bisection_error:
        membership[0] = (2*bisection_error - x) / bisection_error

    if bisection_error < x and x <= 2*bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2*bisection_error < x and x <= 3*bisection_error:
        membership[1] = (3*bisection_error - x) / bisection_error

    if 2*bisection_error < x and x <= 3*bisection_error:
        membership[2] = (x - 2*bisection_error) / bisection_error
    elif 3*bisection_error < x and x <= 4*bisection_error:
        membership[2] = (4*bisection_error - x) / bisection_error

    if 3*bisection_error < x and x <= 4*bisection_error:
        membership[3] = (x - 3*bisection_error) / bisection_error
    elif 4*bisection_error < x and x <= 5*bisection_error:
        membership[3] = (5*bisection_error - x) / bisection_error

    if 4*bisection_error < x and x <= 5*bisection_error:
        membership[4] = (x - 4*bisection_error) / bisection_error
    elif 5*bisection_error < x and x <= 6*bisection_error:
        membership[4] = (6*bisection_error - x) / bisection_error

    if 5*bisection_error < x and x <= 6*bisection_error:
        membership[5] = (x - 5*bisection_error) / bisection_error
    elif 6*bisection_error < x and x <= 7*bisection_error:
        membership[5] = (7*bisection_error - x) / bisection_error

    if 6*bisection_error < x and x <= 7*bisection_error:
        membership[6] = (x - 6*bisection_error) / bisection_error
    elif 7*bisection_error < x and x <= 8*bisection_error:
        membership[6] = (8*bisection_error - x) / bisection_error

    if 7*bisection_error < x and x <= 8*bisection_error:
        membership[7] = (x - 7*bisection_error) / bisection_error
    elif 8*bisection_error < x and x <= 9*bisection_error:
        membership[7] = (9*bisection_error - x) / bisection_error

    if 8*bisection_error < x and x <= 9*bisection_error:
        membership[8] = (x - 8*bisection_error) / bisection_error
    elif 8*bisection_error <= x:
        membership[8] = 1
    return membership


def fuzzyP11(x, max_err):
    membership = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1 / 11 * max_err

    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2 * bisection_error:
        membership[0] = (2 * bisection_error - x) / bisection_error

    if bisection_error < x and x <= 2 * bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[1] = (3 * bisection_error - x) / bisection_error

    if 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[2] = (x - 2 * bisection_error) / bisection_error
    elif 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[2] = (4 * bisection_error - x) / bisection_error

    if 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[3] = (x - 3 * bisection_error) / bisection_error
    elif 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[3] = (5 * bisection_error - x) / bisection_error

    if 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[4] = (x - 4 * bisection_error) / bisection_error
    elif 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[4] = (6 * bisection_error - x) / bisection_error

    if 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[5] = (x - 5 * bisection_error) / bisection_error
    elif 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[5] = (7 * bisection_error - x) / bisection_error

    if 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[6] = (x - 6 * bisection_error) / bisection_error
    elif 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[6] = (8 * bisection_error - x) / bisection_error

    if 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[7] = (x - 7 * bisection_error) / bisection_error
    elif 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[7] = (9 * bisection_error - x) / bisection_error

    if 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[8] = (x - 8 * bisection_error) / bisection_error
    elif 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[8] = (10 * bisection_error - x) / bisection_error

    if 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[9] = (x - 9 * bisection_error) / bisection_error
    elif 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[9] = (11 * bisection_error - x) / bisection_error

    if 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[10] = (x - 10 * bisection_error) / bisection_error
    elif 10 * bisection_error <= x:
        membership[10] = 1
    return membership






def fuzzyP13(x, max_err):
    membership = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1 / 13 * max_err

    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2 * bisection_error:
        membership[0] = (2 * bisection_error - x) / bisection_error

    if bisection_error < x and x <= 2 * bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[1] = (3 * bisection_error - x) / bisection_error

    if 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[2] = (x - 2 * bisection_error) / bisection_error
    elif 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[2] = (4 * bisection_error - x) / bisection_error

    if 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[3] = (x - 3 * bisection_error) / bisection_error
    elif 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[3] = (5 * bisection_error - x) / bisection_error

    if 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[4] = (x - 4 * bisection_error) / bisection_error
    elif 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[4] = (6 * bisection_error - x) / bisection_error

    if 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[5] = (x - 5 * bisection_error) / bisection_error
    elif 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[5] = (7 * bisection_error - x) / bisection_error

    if 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[6] = (x - 6 * bisection_error) / bisection_error
    elif 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[6] = (8 * bisection_error - x) / bisection_error

    if 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[7] = (x - 7 * bisection_error) / bisection_error
    elif 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[7] = (9 * bisection_error - x) / bisection_error

    if 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[8] = (x - 8 * bisection_error) / bisection_error
    elif 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[8] = (10 * bisection_error - x) / bisection_error

    if 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[9] = (x - 9 * bisection_error) / bisection_error
    elif 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[9] = (11 * bisection_error - x) / bisection_error

    if 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[10] = (x - 10 * bisection_error) / bisection_error
    elif 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[10] = (12 * bisection_error - x) / bisection_error

    if 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[11] = (x - 11 * bisection_error) / bisection_error
    elif 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[11] = (13 * bisection_error - x) / bisection_error

    if 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[12] = (x - 12 * bisection_error) / bisection_error
    elif 12 * bisection_error <= x:
        membership[12] = 1
    return membership







def fuzzyP15(x, max_err):
    membership = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1 / 15 * max_err

    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2 * bisection_error:
        membership[0] = (2 * bisection_error - x) / bisection_error

    if bisection_error < x and x <= 2 * bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[1] = (3 * bisection_error - x) / bisection_error

    if 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[2] = (x - 2 * bisection_error) / bisection_error
    elif 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[2] = (4 * bisection_error - x) / bisection_error

    if 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[3] = (x - 3 * bisection_error) / bisection_error
    elif 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[3] = (5 * bisection_error - x) / bisection_error

    if 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[4] = (x - 4 * bisection_error) / bisection_error
    elif 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[4] = (6 * bisection_error - x) / bisection_error

    if 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[5] = (x - 5 * bisection_error) / bisection_error
    elif 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[5] = (7 * bisection_error - x) / bisection_error

    if 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[6] = (x - 6 * bisection_error) / bisection_error
    elif 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[6] = (8 * bisection_error - x) / bisection_error

    if 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[7] = (x - 7 * bisection_error) / bisection_error
    elif 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[7] = (9 * bisection_error - x) / bisection_error

    if 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[8] = (x - 8 * bisection_error) / bisection_error
    elif 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[8] = (10 * bisection_error - x) / bisection_error

    if 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[9] = (x - 9 * bisection_error) / bisection_error
    elif 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[9] = (11 * bisection_error - x) / bisection_error

    if 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[10] = (x - 10 * bisection_error) / bisection_error
    elif 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[10] = (12 * bisection_error - x) / bisection_error

    if 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[11] = (x - 11 * bisection_error) / bisection_error
    elif 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[11] = (13 * bisection_error - x) / bisection_error

    if 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[12] = (x - 12 * bisection_error) / bisection_error
    elif 13 * bisection_error < x and x <= 14 * bisection_error:
        membership[12] = (14 * bisection_error - x) / bisection_error

    if 13 * bisection_error < x and x <= 14 * bisection_error:
        membership[13] = (x - 13 * bisection_error) / bisection_error
    elif 14 * bisection_error < x and x <= 15 * bisection_error:
        membership[13] = (15 * bisection_error - x) / bisection_error

    if 14 * bisection_error < x and x <= 15 * bisection_error:
        membership[14] = (x - 14 * bisection_error) / bisection_error
    elif 14 * bisection_error <= x:
        membership[14] = 1
    return membership












def fuzzyP17(x, max_err):
    membership = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1 / 17 * max_err

    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2 * bisection_error:
        membership[0] = (2 * bisection_error - x) / bisection_error

    if bisection_error < x and x <= 2 * bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[1] = (3 * bisection_error - x) / bisection_error

    if 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[2] = (x - 2 * bisection_error) / bisection_error
    elif 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[2] = (4 * bisection_error - x) / bisection_error

    if 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[3] = (x - 3 * bisection_error) / bisection_error
    elif 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[3] = (5 * bisection_error - x) / bisection_error

    if 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[4] = (x - 4 * bisection_error) / bisection_error
    elif 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[4] = (6 * bisection_error - x) / bisection_error

    if 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[5] = (x - 5 * bisection_error) / bisection_error
    elif 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[5] = (7 * bisection_error - x) / bisection_error

    if 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[6] = (x - 6 * bisection_error) / bisection_error
    elif 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[6] = (8 * bisection_error - x) / bisection_error

    if 7 * bisection_error < x and x <= 8 * bisection_error:
        membership[7] = (x - 7 * bisection_error) / bisection_error
    elif 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[7] = (9 * bisection_error - x) / bisection_error

    if 8 * bisection_error < x and x <= 9 * bisection_error:
        membership[8] = (x - 8 * bisection_error) / bisection_error
    elif 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[8] = (10 * bisection_error - x) / bisection_error

    if 9 * bisection_error < x and x <= 10 * bisection_error:
        membership[9] = (x - 9 * bisection_error) / bisection_error
    elif 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[9] = (11 * bisection_error - x) / bisection_error

    if 10 * bisection_error < x and x <= 11 * bisection_error:
        membership[10] = (x - 10 * bisection_error) / bisection_error
    elif 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[10] = (12 * bisection_error - x) / bisection_error

    if 11 * bisection_error < x and x <= 12 * bisection_error:
        membership[11] = (x - 11 * bisection_error) / bisection_error
    elif 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[11] = (13 * bisection_error - x) / bisection_error

    if 12 * bisection_error < x and x <= 13 * bisection_error:
        membership[12] = (x - 12 * bisection_error) / bisection_error
    elif 13 * bisection_error < x and x <= 14 * bisection_error:
        membership[12] = (14 * bisection_error - x) / bisection_error

    if 13 * bisection_error < x and x <= 14 * bisection_error:
        membership[13] = (x - 13 * bisection_error) / bisection_error
    elif 14 * bisection_error < x and x <= 15 * bisection_error:
        membership[13] = (15 * bisection_error - x) / bisection_error

    if 14 * bisection_error < x and x <= 15 * bisection_error:
        membership[14] = (x - 14 * bisection_error) / bisection_error
    elif 15 * bisection_error < x and x <= 16 * bisection_error:
        membership[14] = (16 * bisection_error - x) / bisection_error

    if 16 * bisection_error < x and x <= 17 * bisection_error:
        membership[16] = (x - 16 * bisection_error) / bisection_error
    elif 16 * bisection_error <= x:
        membership[16] = 1
    return membership