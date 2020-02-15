clc;
clear;
close all;
if ~exist('sgwt','dir')
    unzip('https://wiki.epfl.ch/sgwt/documents/sgwt_toolbox-1.02.zip','sgwt')
end
run 'sgwt/sgwt_toolbox/sgwt_setpath.m'

addpath(pwd);

addpath("mask_distances/")
