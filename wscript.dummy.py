import maf
import maflib.util
import sys
sys.path.append('build/')
from parameters import dataset_file, tagsize
import json

def configure(conf):
    pass

def build(exp):
    experiment(exp)

def experiment(exp):
    # if you want to change NUM_FOLD, you need to remove build/train and build/test files
    #params = json.load(open('settings/test.json'))
    params = dict(dataset = 'vine',
                  label = 'favs',
                  rank_method = ['p0'],
                  fold = 5,
                  toptagsize = [100],
                  classification = 0,
                  rate = 30, # parameter for classification
                  powers = ['1.0 1.0']
                  # powers = ['1.0 1.0', '1.0 1.2', '1.0 1.4', '1.0 1.6', '1.0 1.8', 
                  #           '1.2 1.0', '1.4 1.0', '1.6 1.0', '1.8 1.0']
    )

    if params['classification']:
        dataset_file_key = '{0}_{1}_{2}'.format(params['dataset'], params['label'], params['rate'])
    else:
        dataset_file_key = params['dataset']
    # n-foldの交差検証のためにデータセットをn通りに分割
    exp(source=dataset_file[dataset_file_key],
        target='train test',
        # parametersを指定することで、パラメータ付けられたタスクや
        # パラメータ付けられた出力ファイルを作ることができる
        parameters=maflib.util.product({
            'fold' : [i for i in xrange(params['fold'])],
            'dataset' : [params['dataset']],
        }),
        # 1行1データの形式のデータセットをn通りのtrain testに分割する
        # 出力されるtrain, testは'fold'パラメータでパラメータ付けられる
        rule='python segment_by_line.py -n %d -f ${fold} -i ${SRC} -o ${TGT}' % params['fold'])
    
    # 分割した各foldに対して実験
    exp(source='train',
        target='weight',
        parameters=maflib.util.product({  # execute all parameter patterns
            'label': [params['label']],
            'rank_method': params['rank_method'],
            'powers': params['powers'],
            'classification' : [params['classification']]
        }),  # 'fold'はもう指定しなくて良い (trainに紐付いている)
        rule='python learn_weight.py -i ${SRC} -o ${TGT} -l ${label} -r ${rank_method} -d ${dataset} -p ${powers} -c ${classification}')

    # ここにはもうfold, param1, param2などのパラメータを指定する必要はない
    # (modelとtestに紐付いているので)
    exp(source='weight train',
        target='model',
        parameters=[{'toptagsize' : i } for i in params['toptagsize']],
        rule='python train.py -i ${SRC} -o ${TGT} -d ${dataset} -l ${label} -r ${rank_method} -tt ${toptagsize} -c ${classification}')
 
    exp(source='weight model test',
        target='result',
        rule='python test.py -i ${SRC} -o ${TGT} -l ${label} -r ${rank_method} -d ${dataset} -tt ${toptagsize} -c ${classification}')

    # foldで分割していた結果を統合. social popularityの実際の値とregressionでの予測値との相関を計算.
    # calcurate accuracy in classification task
    exp(source='result',
        target='correlation',
        aggregate_by=['fold'],
        rule='python calc_accuracy.py -i ${SRC} -o ${TGT} -tt ${toptagsize} -l ${label} -r ${rank_method} -d ${dataset} -p ${powers} -c ${classification}')
    
    # if not params['classification']:
    #     exp(source='result',
    #         target='distribution.pdf',
    #         aggregate_by=['fold'],
    #         rule='python get_distribution.py -i ${SRC} -o ${TGT} -tt ${toptagsize}')

    # exp(source='result test',
    #     target='rich_result',
    #     rule='python identify_result.py -i ${SRC} -o ${TGT} -l ${label} -d ${dataset}')

    # exp(source='rich_result',
    #     target='numtag',
    #     aggregate_by=['fold'],
    #     rule='python merge_rich_results.py -i ${SRC} -o ${TGT} -l ${label} -d ${dataset}')

    # aggregate result
    exp(source='correlation',
        target='graph',
        aggregate_by=['toptagsize'],
        rule='python get_json_for_graph.py -i ${SRC} -o ${TGT}')
    
    # get top n (default : n = 1000) tags and its weights.
    exp(source='weight',
        target='toptag',
        aggregate_by=['fold'],
        rule='python get_top_tag.py -i ${SRC} -o ${TGT} -d ${dataset}')

    # get tag collaboration heatmaps similar to confusion matrix
    # exp(source='toptag',
    #     target='toptag_collabo.pdf',
    #     rule='python get_collaboration_images.py -i ${SRC} -o ${TGT} -d ${dataset}')
    
    # merge weight of selected features
    # exp((source='model',
    #      target='selected_weight',
    #      aggregate_by=['fold'],
    #      rule='python get_selected_weight.py -i ${SRC} -o ${TGT} -d ${dataset}')
