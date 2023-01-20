import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor

from .Evaluator import Evaluator


logger = logging.getLogger(__name__)

DEFINITIONS = {
    'hkunlp/instructor-base':{
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying if the sentence is positive or not: ',
        'TweetSentimentExtractionClassification': 'Represent a Twitter sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent an amazon sentence for classifying whether the sentence is counterfactual or not: ',
        'ImdbClassification': 'Represent an amazon review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'represent the sentence for classifying its purpose asweather_query, '
                                         'audio_volume_other, iot_cleaning, '
                                         'datetime_query, lists_remove, lists_createoradd, '
                                         'datetime_convert, play_music, iot_hue_lightdim, '
                                         'calendar_remove, iot_coffee, '
                                         'general_greet, alarm_query, calendar_set, '
                                         'recommendation_locations, '
                                         'lists_query or email_query: ',
        'MassiveScenarioClassification':  "Represent a scene for determining the scene as"
                                         "play, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "email, cooking, news or question answering: ",
        'MTOPDomainClassification': 'represent a sentence; ',
        'MTOPIntentClassification': 'Represent the sentence for determining its purpose as question_music, '
                                    'start_shuffle_music, get_call_time, '
                                    'get_reminder_location, is_true_recipes, '
                                    'ignore_call, get_contact_method, '
                                    'update_reminder, delete_alarm, set_default_provider_music, '
                                    'end_call, '
                                    'skip_track_music, create_timer, cancel_message, '
                                    'get_category_event, repeat_all_off_music, get_timer, '
                                    'add_time_timer, resume_music, add_to_playlist_music, update_reminder_location, '
                                    'set_rsvp_interested, pause_timer, update_timer, play_media, replay_music: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification': 'represent an amazon review sentence: ',
    },
    'hkunlp/instructor-large':{
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying the emotion: ',
        'TweetSentimentExtractionClassification': 'Represent a Tweet sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent a counter-factual sentence for classification: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for classifying the purpose as one of qa_maths, takeaway_order, weather_query, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                       'iot_hue_lighton, iot_hue_lightchange, alarm_remove, play_radio, '
                                       'transport_taxi, datetime_query, lists_remove, lists_createoradd, '
                                       'datetime_convert, play_music, iot_hue_lightdim, email_querycontact, qa_factoid, '
                                       'cooking_query, music_query, qa_currency, calendar_query, music_settings, '
                                       'music_dislikeness, audio_volume_mute, cooking_recipe, general_joke, play_game, '
                                       'news_query, recommendation_events, music_likeness, audio_volume_down, '
                                       'calendar_remove, iot_coffee, transport_traffic, iot_wemo_off, email_sendemail, '
                                       'iot_hue_lightup, social_query, social_post, iot_hue_lightoff, transport_query, '
                                       'general_greet, play_podcasts, alarm_query, calendar_set, alarm_set, '
                                       'transport_ticket, general_quirky, audio_volume_up, iot_wemo_on, qa_definition, '
                                       'recommendation_locations, play_audiobook, email_addcontact, takeaway_query, '
                                       'lists_query or email_query: ',
        'MassiveScenarioClassification': "Represent the scene for classifying its scene as one of calendar, "
                                         "play, general, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "social, lists, email, transport, cooking, weather, news or qa: ",
        'MTOPDomainClassification': 'Represent a sentence: ',
        'MTOPIntentClassification': 'Represent the sentence for retrieval: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification 42.12': 'Represent a review sentence for classification: ',
        'AmazonReviewsClassification': 'Represent a review for classification: ',
    },
    'hkunlp/instructor-xl': {
        'Banking77Classification': 'Represent the bank77 purposes for retrieving its bank intent: ',
        'EmotionClassification':  'Represent the amazon emotion sentence for classifying the emotion: ',
        'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        # 'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for categorizing its task intent as qa_maths, takeaway_order, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                      'or recommendation_locations: ',
        'MassiveScenarioClassification': "represent an ms sentence for retrieving its intent: ",
        'MTOPDomainClassification': 'represent a MTO sentence to retrieve the task intention: ',
        'MTOPIntentClassification': 'Represent an mto sentence for retrieving its behind task intention: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        # 'ToxicConversationsClassification': 'Represent toxicity comments for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        # 'AmazonPolarityClassification': 'Represent the sentiment comment to retrieve a duplicate sentence: ',
        'AmazonReviewsClassification': 'Represent an amazon review sentence to find the emoation; ',
        # 'AmazonReviewsClassification': 'Represent an amazon movie review sentence to categorize the emotion; ',
        'TweetSentimentExtractionClassification': 'Represent Daily-life spoken sentences for categorization; Input: ',
        # 'TweetSentimentExtractionClassification': 'Represent Daily-life spoken expression for classification; Input: ',
    },
}

class kNNClassificationEvaluator(Evaluator):
    def __init__(
        self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        print('use kNNClassificationEvaluator')
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean"]:  # TODO: "dot"
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            ap = average_precision_score(self.y_test, y_pred)
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            scores["ap_" + metric] = ap
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        scores["ap"] = max_ap
        return scores, test_cache


class kNNClassificationEvaluatorPytorch(Evaluator):
    def __init__(
        self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]

        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        print('use kNNClassificationEvaluatorPytorch')
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean", "dot"]:
            if metric == "cosine":
                distances = 1 - self._cos_sim(X_test, X_train)
            elif metric == "euclidean":
                distances = self._euclidean_dist(X_test, X_train)
            elif metric == "dot":
                distances = -self._dot_score(X_test, X_train)
            neigh_indices = torch.topk(distances, k=self.k, dim=1, largest=False).indices
            y_train = torch.tensor(self.y_train)
            y_pred = torch.mode(y_train[neigh_indices], dim=1).values  # TODO: case where there is no majority
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            ap = average_precision_score(self.y_test, y_pred)
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            scores["ap_" + metric] = ap
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        scores["ap"] = max_ap
        return scores, test_cache

    @staticmethod
    def _cos_sim(a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euclidean_dist(a: Tensor, b: Tensor):
        """
        Computes the euclidean distance euclidean_dist(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = euclidean_dist(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.cdist(a, b, p=2)

    @staticmethod
    def _dot_score(a: Tensor, b: Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))


class logRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        sentences_train,
        y_train,
        sentences_test,
        y_test,
        max_iter=100,
        batch_size=32,
        limit=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test
        self.args = kwargs['args']

        self.max_iter = max_iter

        if self.args.batch_size>0:
            self.batch_size = self.args.batch_size
        else:
            self.batch_size = batch_size

    def __call__(self, model, test_cache=None):
        print('use logRegClassificationEvaluator')
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        logger.info(f"Encoding {len(self.sentences_train)} training sentences...")


        if self.args.prompt:
            new_sentences = []
            print('with prompt')
            for s in self.sentences_train:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences_train = new_sentences

            new_sentences = []
            print('with prompt')
            for s in self.sentences_test:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences_test = new_sentences

        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        logger.info(f"Encoding {len(self.sentences_test)} test sentences...")
        # if test_cache is None:
        X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
        test_cache = X_test
        # else:
        #     X_test = test_cache
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        scores["accuracy"] = accuracy
        scores["f1"] = f1

        # if binary classification
        if len(np.unique(self.y_train)) == 2:
            ap = average_precision_score(self.y_test, y_pred)
            scores["ap"] = ap

        return scores, test_cache
