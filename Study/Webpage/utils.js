"use strict";

var utils = {} || utils;

utils.flipCoin = function() {
    return Math.floor((Math.random() * 2));
};

utils.range = function(n) {
    return [...Array(n).keys()]
};

utils.shuffle = function(a) {
    for (var i = a.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }

    return a;
};