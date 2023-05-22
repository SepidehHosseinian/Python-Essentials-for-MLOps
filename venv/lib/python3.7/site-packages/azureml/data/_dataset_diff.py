# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import copy
import json

from enum import Enum

default_diff_config = {"TopN": 10,
                       "enable_diff_calculation": True,
                       "enable_drift_calculation": True,
                       "enable_schema_diff_calculation": True}


class DataSetDiffConstants:
    DIMENSION_COLUMN_NAME = "column_name"
    DIMENSION_METRICS_TYPE = "column_metrics_type"
    DIMENSION_CATEGORY_NAME = "category"
    DIMENSION_PROFILE_DIFF = "profile_diff"
    DIMENSION_SCHEMA_DIFF = "schema_diff"
    DIMENSION_DATASET_DRIFT = "dataset_drift"
    DIMENSION_STATISTICAL_DISTANCE = "statistical_distance"
    DIMENSION_CATEGORICAL_DISTANCE = "categorical_distribution_distance"
    DIMENSION_METRIC_CATEGORY_NAME = "metric_category"
    COMMON_ROW_COUNT = "row_count"
    COMMON_MISSING_COUNT = "missing_count"
    NUMERICAL_PREFIX_PERCENTAGE_DIFFERENCE = "percentage_difference"
    NUMERICAL_PREFIX_DIFFERENCE = "difference"
    NUMERICAL_STATISTICAL_DISTANCE_WASSERSTEIN = "wasserstein_distance"
    NUMERICAL_STATISTICAL_DISTANCE_ENERGY = "energy_distance"
    NUMERICAL_MIN = "min"
    NUMERICAL_MAX = "max"
    NUMERICAL_MEDIAN = "median"
    NUMERICAL_MEAN = "mean"
    NUMERICAL_VARIANCE = "variance"
    NUMERICAL_STANDARD_DEVIATION = "standard_deviation"
    NUMERICAL_SKEWNESS = "skewness"
    NUMERICAL_KURTOSIS = "kurtosis"
    NUMERICAL_PERCENTILES_LIST = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    NUMERICAL_PREFIX_PERCENTILES = "percentile"
    CATEGORICAL_RANK = "rank"
    CATEGORICAL_FREQUENCY = "frequency"
    CATEGORICAL_DISTINCT_VALUE_COUNT = "distinct_value_counts"
    CATEGORICAL_DISTANCE_EUCLIDEAN = "euclidean_distance"
    DRIFT_COEFFICIENT = "datadrift_coefficient"
    DRIFT_CONTRIBUTION = "datadrift_contribution"
    SCHEMA_COMMON_COLUMNS = "common_columns"
    SCHEMA_MISMATCH_COLUMNS = "mismatched_columns"
    SCHEMA_MISMATCH_DTYPE = "mismatched_datatype_columns"


constants = DataSetDiffConstants


class _ColumnType(Enum):
    """Class represents a column type enum."""

    numerical = 1
    categorical = 2
    unknown = 3


class DiffMetric:
    """Class represents a metric."""

    def __init__(self, name, value, extended_properties):
        """Metric constructor.

        :param name: Name of metric
        :type name: str
        :param value: Value of metric
        :type value: float or None
        :param extended_properties: dictionary of string to python primitive type (int, float, str, bool, NoneType)
        :type extended_properties: dict
        :return: A DiffMetric object
        :rtype: DiffMetric
        """
        if name is None or extended_properties is None:
            raise ValueError("name, value and extended_properties must not be None")

        if not isinstance(extended_properties, dict):
            raise TypeError("extended_properties must be dictionary type")

        self.name = str(name)
        self.value = value
        if value is not None:
            self.value = float(value)
        self.extended_properties = extended_properties

    def get_extended_properties(self):
        return copy.deepcopy(self.extended_properties)

    def add_extended_properties(self, ep):
        if not isinstance(ep, dict):
            raise TypeError("extended_properties must be dictionary type")
        self.extended_properties.update(ep)

    def __str__(self):
        ep_dict = copy.deepcopy(self.extended_properties)
        ep_dict["name"] = self.name
        ep_dict["value"] = self.value
        return json.dumps(ep_dict)

    def __repr__(self):
        return self.__str__()

    def _to_json(self):
        return self.__str__()

    @staticmethod
    def _from_json(json_string):
        metric_dict = json.loads(json_string)
        name = metric_dict.pop("name")
        value = metric_dict.pop("value")
        return DiffMetric(name, value, metric_dict)

    @staticmethod
    def _list_to_json(m_list):
        mdict_list = []
        for m in m_list:
            tmp = m.get_extended_properties()
            tmp.update((
                {"name": m.name,
                 "value": m.value}))
            mdict_list.append(tmp)
        return json.dumps(mdict_list)

    @staticmethod
    def _list_from_json(json_string):
        mdict_list = json.loads(json_string)
        results = []
        for mdict in mdict_list:
            name = mdict.pop("name")
            value = mdict.pop("value")
            results.append(DiffMetric(name, value, mdict))
        return results


class ColumnSummary:
    def __init__(self, name, total_ct, distinct_ct, valid_ct, invalid_ct, inferred_dtype, pd_dtype):
        self.name = name
        self.total_row_count = total_ct
        self.distinct_count = distinct_ct
        self.valid_row_count = valid_ct
        self.invalid_row_count = invalid_ct
        self.inferred_dtype = inferred_dtype
        self.pd_dtype = pd_dtype


def get_dataprofile_metrics(dp, config):
    column_type_classifier = {(True, True): _ColumnType.unknown,
                              (True, False): _ColumnType.numerical,
                              (False, True): _ColumnType.categorical,
                              (False, False): _ColumnType.numerical}

    extract_metric_method_selector = {_ColumnType.numerical: get_numerical_column_metrics,
                                      _ColumnType.categorical: get_categorical_column_metrics}

    metrics = {}

    for c in config["columns"]:
        column_type = column_type_classifier[(dp.columns[c].value_counts is None, dp.columns[c].histogram is None)]
        if column_type == _ColumnType.unknown:
            continue
        metrics[c] = extract_metric_method_selector[column_type](dp.columns[c])

    return metrics


def get_numerical_column_metrics(cdp):
    percentiles = constants.NUMERICAL_PERCENTILES_LIST

    metrics = {constants.NUMERICAL_MIN: cdp.min,
               constants.NUMERICAL_MAX: cdp.max,
               constants.NUMERICAL_MEDIAN: cdp.median,
               constants.COMMON_ROW_COUNT: cdp.count,
               constants.COMMON_MISSING_COUNT: cdp.missing_count,
               constants.DIMENSION_COLUMN_NAME: cdp.column_name,
               constants.DIMENSION_METRICS_TYPE: _ColumnType.numerical.value,
               constants.NUMERICAL_KURTOSIS: cdp.moments.kurtosis,
               constants.NUMERICAL_MEAN: cdp.moments.mean,
               constants.NUMERICAL_SKEWNESS: cdp.moments.skewness,
               constants.NUMERICAL_STANDARD_DEVIATION: cdp.moments.standard_deviation,
               constants.NUMERICAL_VARIANCE: cdp.moments.variance}

    for pctile in percentiles:
        metrics["{}_{}".format(
            constants.NUMERICAL_PREFIX_PERCENTILES,
            pctile)] = cdp.quantiles[pctile]

    return metrics


def get_categorical_column_metrics(cdp):
    metrics = {constants.DIMENSION_CATEGORY_NAME: list(),
               constants.CATEGORICAL_FREQUENCY: list(),
               constants.CATEGORICAL_RANK: list(),
               constants.CATEGORICAL_DISTINCT_VALUE_COUNT: cdp.unique_values,
               constants.COMMON_ROW_COUNT: cdp.count,
               constants.COMMON_MISSING_COUNT: cdp.missing_count,
               constants.DIMENSION_COLUMN_NAME: cdp.column_name,
               constants.DIMENSION_METRICS_TYPE: _ColumnType.categorical.value}

    for i in range(len(cdp.value_counts)):
        metrics[constants.DIMENSION_CATEGORY_NAME].append(str(cdp.value_counts[i].value))
        metrics[constants.CATEGORICAL_FREQUENCY].append(cdp.value_counts[i].count)
        metrics[constants.CATEGORICAL_RANK].append(i + 1)

    return metrics


def get_config(config, name):
    if config is None:
        config = copy.deepcopy(default_diff_config)

    return config[name]


def get_numerical_distribution_from_dataprofile(profile):
    x = []
    x_weight = []

    for bucket in profile.histogram:
        x.append((bucket.upper_bound + bucket.lower_bound) / 2)
        x_weight.append(bucket.count)

    return x, x_weight


class PdUtils:
    PD_INVALID_PERCENT_THRESHOLD = 0.15
    PD_CATEGORICAL_CARD_MAX_PERCENT_TOTAL = 0.05
    PD_CATEGORICAL_CARD_MAX = 100

    def __init__(self):
        import pandas
        self.pd = pandas

    def get_inferred_categorical_columns(self, summaries):
        return [c.name for c in summaries
                if self.pd.api.types.is_categorical_dtype(c.inferred_dtype)]

    def get_inferred_fillna_columns(self, summaries):
        return [c.name for c in summaries
                if self.pd.api.types.is_numeric_dtype(c.inferred_dtype)
                and c.invalid_row_count != 0
                and c.invalid_row_count / c.total_row_count < PdUtils.PD_INVALID_PERCENT_THRESHOLD]

    def get_inferred_drop_columns(self, summaries):
        return [c.name for c in summaries
                if (self.pd.api.types.is_numeric_dtype(c.inferred_dtype)
                    and c.invalid_row_count / c.total_row_count >= PdUtils.PD_INVALID_PERCENT_THRESHOLD)
                or (not self.pd.api.types.is_numeric_dtype(c.inferred_dtype) and not
                    self.pd.api.types.is_categorical_dtype(c.inferred_dtype))]

    def get_pandas_df_summary(self, df):
        total_row_counts = len(df.index)
        max_cat_set_size = min(int(total_row_counts * PdUtils.PD_CATEGORICAL_CARD_MAX_PERCENT_TOTAL),
                               PdUtils.PD_CATEGORICAL_CARD_MAX)
        valid_counts_dict = df.count().to_dict()
        dtypes_dict = PdUtils._get_dtypes_dict(df)

        summaries = []
        for col in df.columns:
            dcounts = len(df[col].unique())

            inferred_dtype = dtypes_dict[col]
            if (self.pd.api.types.is_numeric_dtype(inferred_dtype)
                or self.pd.api.types.is_string_dtype(inferred_dtype)) \
                    and dcounts <= max_cat_set_size:
                inferred_dtype = self.pd.api.types.CategoricalDtype()

            summaries.append(ColumnSummary(
                name=col,
                total_ct=total_row_counts,
                distinct_ct=dcounts,
                valid_ct=valid_counts_dict[col],
                invalid_ct=total_row_counts - valid_counts_dict[col],
                inferred_dtype=inferred_dtype,
                pd_dtype=dtypes_dict[col]))

        return summaries

    @staticmethod
    def _get_dtypes_dict(df):
        dtypes_dict = {}
        for c, t in df.dtypes.iteritems():
            dtypes_dict[c] = t

        return dtypes_dict


class ClassifierDriftDetector:
    def __init__(self, base_dataset, diff_dataset):
        # verify input dataset has same columns
        if base_dataset is None or diff_dataset is None:
            raise ValueError("Inputs cannot be None")

        import pandas
        import numpy
        import lightgbm
        import sklearn
        self.pd = pandas
        self.np = numpy
        self.lightgbm = lightgbm
        self.sklearn = sklearn

        self.base_dataset = base_dataset
        self.diff_dataset = diff_dataset
        self.model_dict = {}
        self.drift_extended_properties = {}

    @staticmethod
    def get_supported_dtypes():
        """Get supported datatypes.

        :return: List of string
        :rtype: list(str)
        """
        return ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'category']

    @staticmethod
    def is_supported_dtype(dtype):
        """Check if dtype is supported.

        :param dtype: dtype of a column
        :type dtype: pandas.DataFrame.dtype
        :return: whether given dtype is supported
        :rtype: boolean
        """
        import pandas as pd
        if pd.api.types.is_numeric_dtype(dtype):
            return True

        if pd.api.types.is_categorical_dtype(dtype):
            return True

        return False

    @staticmethod
    def _is_dataframe_dtypes_supported(df):
        supported = True
        for c in df.dtypes:
            if not ClassifierDriftDetector.is_supported_dtype(c):
                supported = False
                break

        return supported

    @staticmethod
    def _is_dataframe_columns_equal(df1, df2):
        column_list1 = []
        column_list2 = []

        for i, v in df1.dtypes.iteritems():
            column_list1.append("{}_{}".format(i, ClassifierDriftDetector._get_generic_type(v)))

        for i, v in df2.dtypes.iteritems():
            column_list2.append("{}_{}".format(i, ClassifierDriftDetector._get_generic_type(v)))

        return set(column_list1) == set(column_list2)

    @staticmethod
    def _get_generic_type(type_string):
        if type_string in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            return "numeric"
        else:
            return type_string

    @staticmethod
    def _is_model_potentially_invalid(pred_output):
        if len(set(pred_output)) == 1:
            return True
        return False

    def get_classifier_drift_detector_results(self, x1, x2, learner=None):
        """Return mcc and feature importance from classifier trained to calssify x1, x2.

        :param x1: Dataframe 1 for diff
        :type x1: pandas.DataFrame
        :param x2: Dataframe 2 for diff
        :type x2: pandas.DataFrame
        :param learner: classifier that supports fit and predict method call
        :type learner: sklearn.classifier
        :return: train_mcc
        :rtype: float
        :return: test_mcc
        :rtype: float
        :return: all_mcc
        :rtype: float
        :return: fea_imp_vec
        :rtype: numpy.array[float]
        """
        if learner is None:
            learner = self.lightgbm.LGBMClassifier

        if not ClassifierDriftDetector._is_dataframe_columns_equal(x1, x2):
            raise AssertionError("Input datasets do not share identical columns")
        if not ClassifierDriftDetector._is_dataframe_dtypes_supported(x1):
            raise AssertionError("Input data contains unsupported datatype column")

        assert x1.shape[1] == x2.shape[1]

        data1 = x1
        data2 = x2

        target1 = self.np.zeros((data1.shape[0], 1))
        target2 = self.np.ones((data2.shape[0], 1))

        data = self.pd.concat([data1, data2], ignore_index=True)
        target = self.np.concatenate((target1, target2), 0)

        sss = self.sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=33
        )

        for train, test in sss.split(data, target):
            clf1 = learner(n_jobs=-1)
            clf1.fit(data.iloc[train], target[train].flatten())

            all_preds = clf1.predict(data)
            all_mcc = self.get_mcc(target, all_preds)

            is_model_potentially_invalid = ClassifierDriftDetector._is_model_potentially_invalid(all_preds)

            test_preds = clf1.predict(data.iloc[test])
            test_mcc = self.get_mcc(
                target[test],
                test_preds)

            train_preds = clf1.predict(data.iloc[train])
            train_mcc = self.get_mcc(
                target[train],
                train_preds)

        fea_imp_dict = {}
        for index in range(len(clf1.feature_importances_)):
            fea_imp_dict[data.columns.values[index]] = clf1.feature_importances_[index]

        return train_mcc, test_mcc, all_mcc, fea_imp_dict, clf1, is_model_potentially_invalid

    def get_mcc(self, y_true, y_pred):
        tn, fp, fn, tp = self.sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        denominator = {"tp_fp": tp + fp,
                       "tp_fn": tp + fn,
                       "tn_fp": tn + fp,
                       "tn_fn": tn + fn}
        if any(m == 0 for m in denominator.values()):
            return 0

        return self.sklearn.metrics.matthews_corrcoef(y_true, y_pred)

    def get_preprocessed_dfs(self, base, diff):
        common_columns = base.columns & diff.columns
        pdutils = PdUtils()

        dfsummaries_base = pdutils.get_pandas_df_summary(base[common_columns])
        dfsummaries_diff = pdutils.get_pandas_df_summary(diff[common_columns])

        columns_to_drop = list(set(pdutils.get_inferred_drop_columns(dfsummaries_base)).union(
            set(pdutils.get_inferred_drop_columns(dfsummaries_diff))))

        base_fillna_columns = pdutils.get_inferred_fillna_columns(dfsummaries_base)
        diff_fillna_columns = pdutils.get_inferred_fillna_columns(dfsummaries_diff)

        common_columns_filtered = list(set(common_columns) - set(columns_to_drop))

        categorical_columns = list(set(pdutils.get_inferred_categorical_columns(dfsummaries_base)).intersection(
            common_columns_filtered))

        self.drift_extended_properties.update({"dropped_columns": columns_to_drop,
                                               "categorical_columns": categorical_columns,
                                               "base_fill_na_columns": base_fillna_columns,
                                               "diff_fill_na_columns": diff_fillna_columns})

        for c in categorical_columns:
            base[c].fillna(value="__null__", inplace=True)
            diff[c].fillna(value="__null__", inplace=True)
            common_cats = list(set(self.np.concatenate((base[c].unique(), diff[c].unique()))))
            cat_type = self.pd.api.types.CategoricalDtype(categories=common_cats)
            base.loc[:, c] = base[c].astype(cat_type)
            diff.loc[:, c] = diff[c].astype(cat_type)

        base[base_fillna_columns] = base[base_fillna_columns].fillna(base[base_fillna_columns].mean())
        # diff fillna with mean using baseline dataset to avoid introducing highly distinguishable values.
        # This is intended and not a bug.
        diff[diff_fillna_columns] = diff[diff_fillna_columns].fillna(base[diff_fillna_columns].mean())

        return base[common_columns_filtered], diff[common_columns_filtered]

    def get_drift_metrics(self, ep={}, include_columns=None):
        base_ds = self.base_dataset
        diff_ds = self.diff_dataset
        common_columns = set(base_ds.columns & diff_ds.columns)

        if include_columns is None or len(include_columns) == 0:
            columns = list(common_columns)
            self.drift_extended_properties.update({"common_columns": columns})
        else:
            columns = list(set(include_columns) & common_columns)
            self.drift_extended_properties.update({"include_columns": include_columns})
            missing_columns = list(set(include_columns) - set(columns))
            if len(missing_columns) != 0:
                self.drift_extended_properties.update({"missing_columns": missing_columns})

        base_processed, diff_processed = self.get_preprocessed_dfs(
            base_ds[columns], diff_ds[columns])

        assert len(base_processed.columns) != 0, "No columns are available for diff calculation."
        self.drift_extended_properties.update({"diff_eligible_columns": list(base_processed.columns)})

        train_mcc, test_mcc, all_mcc, feature_importances, model, potentially_invalid = \
            self.get_classifier_drift_detector_results(base_processed, diff_processed)

        self.drift_extended_properties.update(
            {
                "model": model,
                "train_mcc": train_mcc,
                "test_mcc": test_mcc,
                "all_mcc": all_mcc
            }
        )

        if potentially_invalid:
            self.drift_extended_properties.update(
                {
                    "is_model_potentially_invalid": True
                }
            )

        results = []

        for key, value in feature_importances.items():
            this_ep = copy.deepcopy(ep)
            this_ep[constants.DIMENSION_COLUMN_NAME] = key
            results.append(DiffMetric(
                name=constants.DRIFT_CONTRIBUTION,
                value=value,
                extended_properties=this_ep
            ))

        results.append(DiffMetric(name=constants.DRIFT_COEFFICIENT,
                                  value=test_mcc,
                                  extended_properties=ep))

        return results


class DatasetDiff:
    def __init__(self, base_dataset, diff_dataset, config=None, ep=None):
        """DatasetDiff constructor.

        :param base_dataset: Dataset
        :param diff_dataset: Dataset
        :param config: Dict<str, obj>
        """
        self.base_dataset = base_dataset
        self.diff_dataset = diff_dataset

        profile_arguments = {
            'number_of_histogram_bins': 100
        }

        self.base_datasetprofile = base_dataset.get_profile(arguments=profile_arguments)
        self.diff_datasetprofile = diff_dataset.get_profile(arguments=profile_arguments)
        self.config = config

        if self.config is None:
            self.config = copy.deepcopy(default_diff_config)

        self.config["columns"] = self.get_diff_columns()

        DatasetDiff.validate_diff_config(self.config)

        self.common_ep = ep
        if self.common_ep is None:
            self.common_ep = {}

    def get_diff_columns(self):
        if "columns" not in self.config.keys():
            common_columns = self.base_datasetprofile.columns.keys() & self.diff_datasetprofile.columns.keys()

            include_not_null = self.config.get("include_columns") is not None
            exclude_not_null = self.config.get("exclude_columns") is not None

            column_selector = {(True, False): common_columns & set(self.config.get("include_columns", set())),
                               (False, True): common_columns - set(self.config.get("exclude_columns", set())),
                               (True, True): common_columns,
                               (False, False): common_columns}

            columns = list(column_selector[(include_not_null, exclude_not_null)])
            return columns

        return self.config["columns"]

    @staticmethod
    def validate_diff_config(config):
        if "columns" not in config.keys():
            raise ValueError("columns is required configuration")
        return True

    @staticmethod
    def get_metric_pct_diff(m1, m2):
        """Get percentage difference of two numerical number.

        :return: float
        :rtype: float
        """
        if m1 == 0:
            return None

        if m1 is None or m2 is None:
            return None

        return float((m2 - m1) / abs(m1))

    @staticmethod
    def get_metric_diff(m1, m2):
        """Get difference of two numerical number.

        :return: float
        :rtype: float
        """
        if m1 is None or m2 is None:
            return None

        return float(m2 - m1)

    @staticmethod
    def get_metric_doc_category(ds1_topn_category, ds2_topn_category):
        """Get degree of change of category.

        :return: float
        :rtype: float
        """
        n = len(ds1_topn_category)

        if n == 0:
            return None

        diffset = set(ds2_topn_category - ds1_topn_category)
        return float(len(diffset) / n)

    @staticmethod
    def _get_unioned_normalized_categorical_distributions(p1, p2):
        p1_distribution = dict(zip(p1[constants.DIMENSION_CATEGORY_NAME], p1[constants.CATEGORICAL_FREQUENCY]))
        p2_distribution = dict(zip(p2[constants.DIMENSION_CATEGORY_NAME], p2[constants.CATEGORICAL_FREQUENCY]))

        unioned_category = set(p1[constants.DIMENSION_CATEGORY_NAME] + p2[constants.DIMENSION_CATEGORY_NAME])

        p1_unioned_distribution = []
        p2_unioned_distribution = []
        for cat in unioned_category:
            p1_unioned_distribution.append(p1_distribution.get(cat, 0))
            p2_unioned_distribution.append(p2_distribution.get(cat, 0))

        p1_unioned_distribution_normalized = [f / sum(p1_unioned_distribution) for f in p1_unioned_distribution]
        p2_unioned_distribution_normalized = [f / sum(p2_unioned_distribution) for f in p2_unioned_distribution]

        ret_p1 = dict(zip(list(unioned_category), p1_unioned_distribution_normalized))
        ret_p2 = dict(zip(list(unioned_category), p2_unioned_distribution_normalized))

        return ret_p1, ret_p2

    @staticmethod
    def _get_numerical_diffs_metrics(m1, m2, metric_name, ep):
        results = []

        num_diff_metric_list = [{"name": constants.NUMERICAL_PREFIX_PERCENTAGE_DIFFERENCE,
                                 "method": DatasetDiff.get_metric_pct_diff},
                                {"name": constants.NUMERICAL_PREFIX_DIFFERENCE,
                                 "method": DatasetDiff.get_metric_diff}
                                ]

        for diff_method in num_diff_metric_list:
            results.append(DiffMetric(name=diff_method["name"] + "_{}".format(metric_name),
                                      value=diff_method["method"](m1, m2),
                                      extended_properties=ep))
        return results

    @staticmethod
    def _get_categorical_by_category_diff_metrics(p1, p2, ep, config):
        results = []

        for cat in p1[constants.DIMENSION_CATEGORY_NAME][:get_config(config, "TopN")]:
            this_ep = copy.deepcopy(ep)
            this_ep[constants.DIMENSION_CATEGORY_NAME] = cat
            for m_name in [constants.CATEGORICAL_RANK,
                           constants.CATEGORICAL_FREQUENCY]:
                m1_cat = DatasetDiff._get_category_metric(cat, p1)
                m2_cat = DatasetDiff._get_category_metric(cat, p2)
                results.extend(DatasetDiff._get_numerical_diffs_metrics(m1=m1_cat[m_name],
                                                                        m2=m2_cat[m_name],
                                                                        metric_name=m_name,
                                                                        ep=this_ep))
        return results

    @staticmethod
    def _get_category_metric(cat, m):
        if cat not in m[constants.DIMENSION_CATEGORY_NAME]:
            return {constants.CATEGORICAL_FREQUENCY: None,
                    constants.CATEGORICAL_RANK: None}

        cat_index = m[constants.DIMENSION_CATEGORY_NAME].index(cat)
        freq = m[constants.CATEGORICAL_FREQUENCY][cat_index]
        rank = m[constants.CATEGORICAL_RANK][cat_index]

        return {constants.CATEGORICAL_FREQUENCY: freq,
                constants.CATEGORICAL_RANK: rank}

    @staticmethod
    def _get_numerical_distance_metric(p1, p2, m_name, ep):
        from scipy.stats import energy_distance, wasserstein_distance

        dm_methods = {constants.NUMERICAL_STATISTICAL_DISTANCE_WASSERSTEIN: wasserstein_distance,
                      constants.NUMERICAL_STATISTICAL_DISTANCE_ENERGY: energy_distance}

        x1, x1_weight = get_numerical_distribution_from_dataprofile(p1)
        x2, x2_weight = get_numerical_distribution_from_dataprofile(p2)

        if all(w == 0 for w in x1_weight) or all(w == 0 for w in x2_weight):
            distance = None
        else:
            distance = dm_methods[m_name](x1, x2, x1_weight, x2_weight)

        results = [DiffMetric(name=m_name,
                              value=distance,
                              extended_properties=ep)]

        return results

    @staticmethod
    def _get_categorical_distance_metric(p1, p2, m_name, ep):
        from scipy.spatial.distance import euclidean

        dm_methods = {constants.CATEGORICAL_DISTANCE_EUCLIDEAN: euclidean}

        x1, x2 = DatasetDiff._get_unioned_normalized_categorical_distributions(p1, p2)
        distance = dm_methods[m_name](list(x1.values()), list(x2.values()))
        results = [DiffMetric(name=m_name,
                              value=distance,
                              extended_properties=ep)]
        return results

    @staticmethod
    def _get_schema_diff_results(df_a, df_b, m_name, ep):
        methods = {constants.SCHEMA_COMMON_COLUMNS: DatasetDiff.get_common_columns,
                   constants.SCHEMA_MISMATCH_COLUMNS: DatasetDiff.get_mismatch_columns,
                   constants.SCHEMA_MISMATCH_DTYPE: DatasetDiff.get_mismatch_dtype_columns}

        this_ep = copy.deepcopy(ep)
        this_ep["result"] = list(methods[m_name](df_a, df_b))
        results = [DiffMetric(name=m_name,
                              value=None,
                              extended_properties=this_ep)]

        return results

    @staticmethod
    def get_common_columns(ds1, ds2):
        return set(ds1.definition.dtypes.keys()) & set(ds2.definition.dtypes.keys())

    @staticmethod
    def get_mismatch_columns(ds1, ds2):
        return set(ds1.definition.dtypes.keys()).symmetric_difference(ds2.definition.dtypes.keys())

    @staticmethod
    def get_mismatch_dtype_columns(ds1, ds2):
        common_columns = list(DatasetDiff.get_common_columns(ds1, ds2))
        cols_w_dtype_a = []
        cols_w_dtype_b = []

        for c in common_columns:
            cols_w_dtype_a.append("{}:{}".format(c, ds1.definition.dtypes[c]))
            cols_w_dtype_b.append("{}:{}".format(c, ds2.definition.dtypes[c]))

        return set(cols_w_dtype_a).symmetric_difference(cols_w_dtype_b)

    @staticmethod
    def get_numerical_only_metrics_list():
        metric_list = [
            constants.NUMERICAL_MIN,
            constants.NUMERICAL_MAX,
            constants.NUMERICAL_MEDIAN,
            constants.NUMERICAL_MEAN,
            constants.NUMERICAL_VARIANCE,
            constants.NUMERICAL_STANDARD_DEVIATION,
            constants.NUMERICAL_SKEWNESS,
            constants.NUMERICAL_KURTOSIS
        ]

        for pctile in constants.NUMERICAL_PERCENTILES_LIST:
            metric_list.append(
                "{}_{}".format(
                    constants.NUMERICAL_PREFIX_PERCENTILES,
                    pctile)
            )
        return metric_list

    @staticmethod
    def get_statistical_distance_metrics_list():
        metric_list = [
            constants.NUMERICAL_STATISTICAL_DISTANCE_WASSERSTEIN,
            constants.NUMERICAL_STATISTICAL_DISTANCE_ENERGY]
        return metric_list

    @staticmethod
    def get_categorical_distance_metrics_list():
        metric_list = [
            constants.CATEGORICAL_DISTANCE_EUCLIDEAN]
        return metric_list

    @staticmethod
    def get_common_metrics_list():
        metric_list = [
            constants.COMMON_ROW_COUNT,
            constants.COMMON_MISSING_COUNT
        ]
        return metric_list

    @staticmethod
    def get_categorical_only_metrics_list():
        metric_list = [
            constants.CATEGORICAL_DISTINCT_VALUE_COUNT
        ]
        return metric_list

    @staticmethod
    def get_categorical_by_category_metrics_list():
        metric_list = [
            constants.CATEGORICAL_RANK,
            constants.CATEGORICAL_FREQUENCY
        ]
        return metric_list

    @staticmethod
    def get_schema_diff_list():
        metric_list = [
            constants.SCHEMA_COMMON_COLUMNS,
            constants.SCHEMA_MISMATCH_COLUMNS,
            constants.SCHEMA_MISMATCH_DTYPE
        ]
        return metric_list

    def run(self):
        """ Return list of metric as specified for computation in config object.

        :return: metrics
        :rtype: list[Metric]
        """

        base_profile_metrics = get_dataprofile_metrics(self.base_datasetprofile, self.config)
        diff_profile_metrics = get_dataprofile_metrics(self.diff_datasetprofile, self.config)

        diff_metrics = []

        common = DatasetDiff.get_common_metrics_list()
        numerical_metrics = DatasetDiff.get_numerical_only_metrics_list()
        categorical_metrics = DatasetDiff.get_categorical_only_metrics_list()
        distance_metrics = DatasetDiff.get_statistical_distance_metrics_list()
        categorical_distance_metrics = DatasetDiff.get_categorical_distance_metrics_list()
        schema_diffs = DatasetDiff.get_schema_diff_list()

        common_ep = self.common_ep

        if self.config.get("enable_diff_calculation", True):
            for c in self.config["columns"]:
                p1 = base_profile_metrics.get(c, None)
                p2 = diff_profile_metrics.get(c, None)

                if p1 is None or p2 is None:
                    continue

                if p1[constants.DIMENSION_METRICS_TYPE] != \
                        p2[constants.DIMENSION_METRICS_TYPE]:
                    continue

                ep = copy.deepcopy(common_ep)
                ep[constants.DIMENSION_COLUMN_NAME] = c
                ep[constants.DIMENSION_METRIC_CATEGORY_NAME] = constants.DIMENSION_PROFILE_DIFF

                if base_profile_metrics[c][constants.DIMENSION_METRICS_TYPE] == \
                        _ColumnType.numerical.value:
                    for m_name in numerical_metrics + common:
                        diff_metrics.extend(
                            DatasetDiff._get_numerical_diffs_metrics(p1[m_name], p2[m_name], m_name, ep))
                    for m_name in distance_metrics:
                        this_ep = copy.deepcopy(ep)
                        this_ep[constants.DIMENSION_METRIC_CATEGORY_NAME] = constants.DIMENSION_STATISTICAL_DISTANCE
                        diff_metrics.extend(
                            DatasetDiff._get_numerical_distance_metric(self.base_datasetprofile.columns[c],
                                                                       self.diff_datasetprofile.columns[c],
                                                                       m_name, this_ep)
                        )

                if base_profile_metrics[c][constants.DIMENSION_METRICS_TYPE] == \
                        _ColumnType.categorical.value:
                    for m_name in categorical_metrics + common:
                        diff_metrics.extend(
                            DatasetDiff._get_numerical_diffs_metrics(p1[m_name], p2[m_name], m_name, ep))
                    for m_name in categorical_distance_metrics:
                        this_ep = copy.deepcopy(ep)
                        this_ep[constants.DIMENSION_METRIC_CATEGORY_NAME] = constants.DIMENSION_CATEGORICAL_DISTANCE
                        diff_metrics.extend(
                            DatasetDiff._get_categorical_distance_metric(p1,
                                                                         p2,
                                                                         m_name, this_ep)
                        )
                    diff_metrics.extend(
                        DatasetDiff._get_categorical_by_category_diff_metrics(p1, p2, ep, self.config)
                    )

        if self.config.get("enable_drift_calculation", True):
            ep = copy.deepcopy(common_ep)
            ep[constants.DIMENSION_METRIC_CATEGORY_NAME] = constants.DIMENSION_DATASET_DRIFT
            dd = ClassifierDriftDetector(self.base_dataset.to_pandas_dataframe(),
                                         self.diff_dataset.to_pandas_dataframe())

            drift_metrics = dd.get_drift_metrics(ep=ep, include_columns=self.config["columns"])
            diff_metrics.extend(drift_metrics)

        if self.config.get("enable_schema_diff_calculation", True):
            ep = copy.deepcopy(common_ep)
            ep[constants.DIMENSION_METRIC_CATEGORY_NAME] = constants.DIMENSION_SCHEMA_DIFF
            for m in schema_diffs:
                diff_metrics.extend(DatasetDiff._get_schema_diff_results(self.base_dataset, self.diff_dataset, m, ep))

        return diff_metrics
