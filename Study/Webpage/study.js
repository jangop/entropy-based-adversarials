"use strict";

var ui = {} || ui;

ui.Controller = function() {
    this.urlBaseImgs = "";
    this.urlSubmitResults = "store_results.php";
    this.timeout = 5000;  // Timeout in milliseconds (e.g. 4s = 4000)

    this.groundTruth = undefined;
    this.results = undefined;
    this.challenges = undefined;
    this.challengesIds = undefined;
    this.curChallengeId = undefined;
    this.curStartTime = undefined;
    this.hTimeout = undefined;

    this.imageLoadedCounter = 0;

    this.initChallenges = function(callback) {
        fetch('batch.json').then(x => x.json()).then(x => {
            this.challenges = x.data;
            this.challengesIds = utils.range(this.challenges.length);

            this.challengesIds = utils.shuffle(this.challengesIds);

            callback();
        })
        .catch(error => console.log(error));
        
        this.results = [];
        this.curChallengeId = 0;
    };

    this.storeResults = function() {
        fetch(this.urlSubmitResults, {
            method : "POST",
            body : JSON.stringify({'data': this.results})
        }).catch(error => console.log(error));
    };

    this.loadGroundTruth = function() {
        fetch('groundTruth.json').then(x => x.json()).then(x => {
            this.groundTruth = x.data;
        })
        .catch(error => console.log(error));
    };

    this.computeScore = function() {
        var identicals_found = 0;
        var adv_found = 0;

        for(var entry in this.results) {
            var i = this.results[entry]["id"];
            var y_pred = this.results[entry]["y"];

            var y = this.groundTruth[i]["image_a_is_original"];

            if(y == -1 && y_pred == 0) {
                identicals_found += 1;
            }
            else if((y == 0 || y == 1) && y_pred == 1) {
                adv_found += 1;
            }
        }
        
        return (identicals_found + adv_found) / (this.groundTruth.length * 1.0);
    };

    this.randomizeButtons = function() {
        if(utils.flipCoin() == true) {
            var btnDifferent = document.getElementById("btnDifferent")
            var btnDifferentInnerHTML = btnDifferent.innerHTML;
            var btnNotDifferent = document.getElementById("btnNotDifferent");
            var btnNotDifferentInnerHTML = btnNotDifferent.innerHTML;

            btnDifferent.id = "btnNotDifferent";
            btnDifferent.innerHTML = btnNotDifferentInnerHTML;
            
            btnNotDifferent.id = "btnDifferent";
            btnNotDifferent.innerHTML = btnDifferentInnerHTML;
        }
    };

    this.enableButtons = function() {
        document.getElementById("btnDifferent").disabled = false;
        document.getElementById("btnNotDifferent").disabled = false;
    };

    this.disableButtons = function() {
        document.getElementById("btnDifferent").disabled = true;
        document.getElementById("btnNotDifferent").disabled = true;

        this.imageLoadedCounter = 0;
    };

    this.makeImagesVisible = function() {
        var imgA = document.getElementById("imgA");
        imgA.style.visibility = "visible";

        var imgB = document.getElementById("imgB");
        imgB.style.visibility = "visible";
    };

    this.makeImagesInvisible = function() {
        var imgA = document.getElementById("imgA");
        imgA.style.visibility = "hidden";

        var imgB = document.getElementById("imgB");
        imgB.style.visibility = "hidden";
    };

    this.onImageLoaded = function() {
        this.imageLoadedCounter += 1;
        
        if(this.imageLoadedCounter == 2) {
            this.enableButtons();

            this.makeImagesVisible();

            this.curStartTime = (new Date()).getTime();

            if(this.timeout != undefined) {
                this.hTimeout = setTimeout(this.makeImagesInvisible.bind(this), this.timeout);
            }
        }
    };

    this.init = function() {
        this.loadGroundTruth();
        this.initChallenges(this.goToNextChallenge.bind(this));

        this.randomizeButtons();

        document.getElementById("btnDifferent").addEventListener("click", this.onClickBtnDifferent.bind(this), false);
        document.getElementById("btnNotDifferent").addEventListener("click", this.onClickBtnNotDifferent.bind(this), false);

        document.getElementById("imgA").addEventListener("load", this.onImageLoaded.bind(this), false);
        document.getElementById("imgB").addEventListener("load", this.onImageLoaded.bind(this), false);
    };

    this.goToNextChallenge = function() {
        clearTimeout(this.hTimeout);
        this.disableButtons();

        if(this.curChallengeId == this.challengesIds.length) {
            this.storeResults();

            var score = this.computeScore();
            console.log(score);
            score = Number(score.toFixed(4));

            alert(`You completed the study with a score of ${score}!\nThank you for participating :)`);
        }
        else {
            var elementsImgs = ["imgA", "imgB"]
            if(utils.flipCoin() == true) {
                elementsImgs = utils.shuffle(elementsImgs);
            }

            this.makeImagesInvisible();

            document.getElementById(elementsImgs[0]).setAttribute("src", this.urlBaseImgs + this.challenges[this.challengesIds[this.curChallengeId]].image_a);
            document.getElementById(elementsImgs[1]).setAttribute("src", this.urlBaseImgs + this.challenges[this.challengesIds[this.curChallengeId]].image_b);

            this.curChallengeId += 1;

            document.getElementById("counter").innerHTML =  this.curChallengeId + "/" + this.challengesIds.length;
        }
    };

    this.onClickBtnDifferent = function() {
        var time = (new Date()).getTime() - this.curStartTime;
        this.results.push({'id': this.challengesIds[this.curChallengeId - 1], 'y': 1, 't': time});

        this.goToNextChallenge();
    };

    this.onClickBtnNotDifferent = function() {
        var time = (new Date()).getTime() - this.curStartTime;
        this.results.push({'id': this.challengesIds[this.curChallengeId - 1], 'y': 0, 't': time});

        this.goToNextChallenge();
    };
};

window.onload = function() {
    new ui.Controller().init();
};
