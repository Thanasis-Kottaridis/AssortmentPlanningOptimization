import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
    Clean Datasets Utils
"""


def addColumnNamesToRawData(df, assortmentProductsCount):
    # declare main column names
    mainColumNames = ["produc_id", "total_sale_avg", "exclusive_sale_avg", "revenue_contribution"]

    # repeat for each assortment products (SOS we use np.tile in order to keep the order of names)
    columnName = np.tile(mainColumNames, assortmentProductsCount)
    # append total revenue column
    columnName = np.append(columnName, ["total_assortment_revenue"])

    df.columns = columnName

    return df


def cleanRawData(df, assortmentProductsCount):
    # copy and drop total assortment renevue
    df_tar = df.total_assortment_revenue
    df.drop("total_assortment_revenue", axis=1, inplace=True)

    ## Make id as index on total_assortment_revenue dataframe
    df_tar = df_tar.to_frame()
    df_tar.index.name = "assorment_id"
    df_tar.reset_index(drop=False, inplace=True)
    df_tar.set_index(["assorment_id", "total_assortment_revenue"], inplace=True)

    # breake df into df list every 3 columns
    dfs = np.array_split(df, assortmentProductsCount, axis=1)

    for df in dfs :
        df.index.name = "assorment_id"
        df.reset_index(drop=False, inplace=True)

    # create clean df
    clean_df = pd.concat(dfs)
    clean_df = clean_df.astype({"produc_id" : int})
    clean_df.set_index(["assorment_id", "produc_id"], inplace=True)
    clean_df.sort_index(inplace=True)

    # merge with total assortment revenue
    clean_df = pd.merge(clean_df, df_tar, left_index=True, right_index=True, how='outer')

    return clean_df


## Get filtered df by item id
def get_assortments_with_item_id(clean_df, item_id):
    filtered_df = clean_df.iloc[clean_df.index.get_level_values('produc_id') == item_id]
    return filtered_df.index.get_level_values('assorment_id')


"""
    Regression DataFrame Builders
"""


def columnNames_builder(total_products, target_column_name):
    columnNames = []
    for i in range(0, total_products):
        columnNames.append("item {}".format(i))

    columnNames.append(target_column_name)
    return columnNames


def df_A_builder(row, total_products, columnNames, targetIndex) :
    items = np.zeros(total_products + 1, np.float)

    ## set 1 to items list for every item in assortment
    for i in range(0, len(row) - 1, 4) :
        item_id = np.int(row[i])
        items[item_id] = 1
        # items[item_id] =  row[-1] / 100 * row[i + 3]

        if item_id == targetIndex :
            ## set last column with assortment mean rev
            items[-1] = np.float(row[-1] / 100 * row[i + 3])

    return list(items)


def df_B_builder(row, total_products, columnNames) :
    items = np.zeros(total_products + 1, np.float)

    ## set 1 to items list for every item in assortment
    for i in range(0, len(row) - 1, 4) :
        item_id = np.int(row[i])
        items[item_id] = 1

    ## set last column with assortment mean rev
    items[-1] = np.float(row[-1])

    return list(items)


"""
    Classification Dataframe Builders
"""


def clean_df_builder(row, total_products, columnNames) :
    items = np.zeros(total_products + 1, np.float)

    ## set 1 to items list for every item in assortment
    for i in range(0, len(row) - 1, 4) :
        item_id = np.int(row[i])
        items[item_id] = row[i + 3]

    ## set last column with assortment mean rev
    items[-1] = np.float(row[-1])

    return list(items)


def mean_item_rev_cont(col) :
    np_col = col.to_numpy()
    np_mean = np.mean(np_col[np_col > 0])
    return np_mean


def clasify_df_A_builder(row, total_products, targetIndex, itemAvg) :
    items = np.zeros(total_products + 1, np.float)

    ## set 1 to items list for every item in assortment
    for i in range(0, len(row) - 1) :
        items[i] = 1 if row[i] > 0 else 0
        # items[item_id] =  row[-1] / 100 * row[i + 3]

        if "item {}".format(i) == targetIndex :
            ## set last column with assortment mean revtarget_item_id
            # items[-1] =  row[i]
            items[-1] = 1 if row[targetIndex] > itemAvg else 0
    return list(items)


def clasify_df_B_builder(row, total_products, targetIndex, itemAvg) :
    items = np.zeros(total_products + 1, np.float)

    ## set 1 to items list for every item in assortment
    for i in range(0, len(row) - 1) :
        items[i] = 1 if row[i] > 0 else 0
        # items[item_id] =  row[-1] / 100 * row[i + 3]

    ## set last column with assortment mean revtarget_item_id
    # items[-1] = row[-1]
    items[-1] = 1 if row[-1] > itemAvg else 0

    return list(items)


"""
    Classification Helper functions 
    for train test split and Kfold
"""


def prepare_df_for_classify(df, avg_series, selected_index=None, numberOfFeatures=17):
    columnNames = columnNames_builder(numberOfFeatures, "label")

    if selected_index == "label":
        test = df.apply( lambda row : clasify_df_B_builder(row, numberOfFeatures, selected_index, avg_series[selected_index]), axis=1)

        # create df in order to predict target_revenue_contribution
        df_result = pd.DataFrame(list(test), columns=columnNames)

        # display(df_result)
        return df_result
    else:
        test = df.apply(lambda row: clasify_df_A_builder(row, numberOfFeatures, selected_index, avg_series[selected_index]), axis=1)

        # create df in order to predict target_revenue_contribution
        df_result = pd.DataFrame(list(test), columns=columnNames)

        # drop selected index if needed
        if selected_index is not None:
            df_result.drop(selected_index, axis=1, inplace=True)

        # display(df_result)
        return df_result


def get_avg_series(df, selected_index):
    # export average for each product and assortment reveniew on test set
    avg_series = df.apply(lambda col: mean_item_rev_cont(col))
    # print (avg_series)

    return avg_series


def normalize_df(df, targetLabel):

    # step 1
    # print(targetLabel)
    label = df[targetLabel].to_numpy()
    # print(label)

    x = df.drop([targetLabel], axis = 1)
    # step 2
    x = StandardScaler().fit_transform(x)  # normalizing the features
    # print("\nShape of normalized data:", x.shape)

    # elenxoume an to kanonikopoiimeno mas data set exei mean 0 kai tipiki apoklisi 1
    # print("\nPrints mean:", np.mean(x), " and Standard deviation of normalized dataset: ", np.std(x))

    # step 3: change feature name
    feature_columns = ['item ' + str(i) for i in range(x.shape[1])]
    normalized_dataFrame = pd.DataFrame(x, columns=feature_columns)

    return normalized_dataFrame, label
