from .dataset      import collate_no_label, load_directory, load_file, load_problem, load_file_spanner
from .supervised   import load_dataset as supervised_load
from .supervised   import collate      as supervised_collate
from .unsupervised import load_dataset as unsupervised_load
from .unsupervised import collate      as unsupervised_collate
from .online       import load_dataset as online_load
from .online       import collate      as online_collate

g_dataset_methods = {
    'supervised_optimal':         (supervised_load, supervised_collate),
    'selfsupervised_optimal':     (unsupervised_load, unsupervised_collate),
    'selfsupervised_suboptimal':  (unsupervised_load, unsupervised_collate),
    'selfsupervised_suboptimal2': (unsupervised_load, unsupervised_collate),
    'unsupervised_optimal':       (unsupervised_load, unsupervised_collate),
    'unsupervised_suboptimal':    (unsupervised_load, unsupervised_collate),
    'online_optimal':             (online_load, online_collate)
}
