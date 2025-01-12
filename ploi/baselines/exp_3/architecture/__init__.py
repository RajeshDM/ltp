from .loss import l1_regularization
from .loss import supervised_optimal_loss, unsupervised_optimal_loss
from .loss import selfsupervised_optimal_loss, selfsupervised_suboptimal_loss, selfsupervised_suboptimal2_loss
from .loss import unsupervised_suboptimal_loss

from .max_base import MaxModelBase, RelationMessagePassingModel as MaxRelationMessagePassingModel
from .add_base import AddModelBase, RelationMessagePassingModel as AddRelationMessagePassingModel
from .max_readout_base import MaxReadoutModelBase, RelationMessagePassingModel as MaxReadoutRelationMessagePassingModel
from .attention_base import AttentionModelBase, RelationMessagePassingModel as AttentionRelationMessagePassingModel
from .add_max_base import AddMaxModelBase, RelationMessagePassingModel as AddMaxRelationMessagePassingModel

# Add models
from .model import SupervisedOptimalAddModel, SelfsupervisedSuboptimalAddModel, SelfsupervisedOptimalAddModel, UnsupervisedOptimalAddModel, UnsupervisedSuboptimalAddModel, OnlineOptimalAddModel
# Max models
from .model import SupervisedOptimalMaxModel, SelfsupervisedSuboptimalMaxModel, SelfsupervisedOptimalMaxModel, UnsupervisedOptimalMaxModel, UnsupervisedSuboptimalMaxModel, OnlineOptimalMaxModel
# Max-readout models
from .model import SupervisedOptimalMaxReadoutModel, SelfsupervisedSuboptimalMaxReadoutModel, SelfsupervisedOptimalMaxReadoutModel, UnsupervisedOptimalMaxReadoutModel, UnsupervisedSuboptimalMaxReadoutModel, OnlineOptimalMaxReadoutModel
# Attention models
from .model import SupervisedOptimalAttentionModel, SelfsupervisedSuboptimalAttentionModel, SelfsupervisedOptimalAttentionModel, UnsupervisedOptimalAttentionModel, UnsupervisedSuboptimalAttentionModel, OnlineOptimalAttentionModel
# Add-max models
from .model import SupervisedOptimalAddMaxModel, SelfsupervisedSuboptimalAddMaxModel, SelfsupervisedOptimalAddMaxModel, UnsupervisedOptimalAddMaxModel, UnsupervisedSuboptimalAddMaxModel, OnlineOptimalAddMaxModel
# Models for new loss
from .model import SelfsupervisedSuboptimalAddModel2, SelfsupervisedSuboptimalMaxModel2, SelfsupervisedSuboptimalAddMaxModel2, SelfsupervisedSuboptimalMaxReadoutModel2

# Settings
from .model import set_max_trace_length
from .loss import set_suboptimal_factor, set_loss_constants

# Maps (aggregation, readout, loss) -> model
g_model_classes = {
    ('max',       True,  'supervised_optimal'):         SupervisedOptimalMaxReadoutModel,
    ('max',       True,  'unsupervised_optimal'):       UnsupervisedOptimalMaxReadoutModel,
    ('max',       True,  'selfsupervised_optimal'):     SelfsupervisedOptimalMaxReadoutModel,
    ('max',       True,  'online_optimal'):             OnlineOptimalMaxReadoutModel,
    ('max',       True,  'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalMaxReadoutModel,
    ('max',       True,  'selfsupervised_suboptimal2'): SelfsupervisedSuboptimalMaxReadoutModel2,
    ('max',       True,  'unsupervised_suboptimal'):    UnsupervisedSuboptimalMaxReadoutModel,
    ('max',       True,  'base'):                       MaxReadoutModelBase,

    ('max',       False, 'supervised_optimal'):         SupervisedOptimalMaxModel,
    ('max',       False, 'unsupervised_optimal'):       UnsupervisedOptimalMaxModel,
    ('max',       False, 'selfsupervised_optimal'):     SelfsupervisedOptimalMaxModel,
    ('max',       False, 'online_optimal'):             OnlineOptimalMaxModel,
    ('max',       False, 'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalMaxModel,
    ('max',       False, 'selfsupervised_suboptimal2'): SelfsupervisedSuboptimalMaxModel2,
    ('max',       False, 'unsupervised_suboptimal'):    UnsupervisedSuboptimalMaxModel,
    ('max',       False, 'base'):                       MaxModelBase,

    ('add',       False, 'supervised_optimal'):         SupervisedOptimalAddModel,
    ('add',       False, 'unsupervised_optimal'):       UnsupervisedOptimalAddModel,
    ('add',       False, 'selfsupervised_optimal'):     SelfsupervisedOptimalAddModel,
    ('add',       False, 'online_optimal'):             OnlineOptimalAddModel,
    ('add',       False, 'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalAddModel,
    ('add',       False, 'selfsupervised_suboptimal2'): SelfsupervisedSuboptimalAddModel2,
    ('add',       False, 'unsupervised_suboptimal'):    UnsupervisedSuboptimalAddModel,
    ('add',       False, 'base'):                       AddModelBase,

    ('addmax',    False, 'supervised_optimal'):         SupervisedOptimalAddMaxModel,
    ('addmax',    False, 'unsupervised_optimal'):       UnsupervisedOptimalAddMaxModel,
    ('addmax',    False, 'selfsupervised_optimal'):     SelfsupervisedOptimalAddMaxModel,
    ('addmax',    False, 'online_optimal'):             OnlineOptimalAddMaxModel,
    ('addmax',    False, 'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalAddMaxModel,
    ('addmax',    False, 'selfsupervised_suboptimal2'): SelfsupervisedSuboptimalAddMaxModel2,
    ('addmax',    False, 'unsupervised_suboptimal'):    UnsupervisedSuboptimalAddMaxModel,
    ('addmax',    False, 'base'):                       AddMaxModelBase,

    ('attention', True,  'supervised_optimal'):         SupervisedOptimalAttentionModel,
    ('attention', True,  'unsupervised_optimal'):       UnsupervisedOptimalAttentionModel,
    ('attention', True,  'selfsupervised_optimal'):     SelfsupervisedOptimalAttentionModel,
    ('attention', True,  'online_optimal'):             OnlineOptimalAttentionModel,
    ('attention', True,  'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalAttentionModel,
    ('attention', True,  'unsupervised_suboptimal'):    UnsupervisedSuboptimalAttentionModel,
    ('attention', True,  'base'):                       AttentionModelBase,

    ('attention', False, 'supervised_optimal'):         SupervisedOptimalAttentionModel,
    ('attention', False, 'unsupervised_optimal'):       UnsupervisedOptimalAttentionModel,
    ('attention', False, 'selfsupervised_optimal'):     SelfsupervisedOptimalAttentionModel,
    ('attention', False, 'online_optimal'):             OnlineOptimalAttentionModel,
    ('attention', False, 'selfsupervised_suboptimal'):  SelfsupervisedSuboptimalAttentionModel,
    ('attention', False, 'unsupervised_suboptimal'):    UnsupervisedSuboptimalAttentionModel,
    ('attention', False, 'base'):                       AttentionModelBase
}
