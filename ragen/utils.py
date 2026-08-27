import random
import numpy as np
from contextlib import contextmanager
from omegaconf import OmegaConf
import dataclasses
import re
from collections.abc import Mapping


_SECRET_CONFIG_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|token|secret|password|credential)",
    re.IGNORECASE,
)


def redact_config(value, key_name: str = ""):
    """Return a log-safe copy of a Hydra config.

    Credentials should normally be supplied through the environment, but this
    helper also protects tracker/console logs when a user accidentally places
    one in a config override.  The input is never mutated.
    """
    from omegaconf import DictConfig, ListConfig

    if _SECRET_CONFIG_KEY.search(key_name):
        if value is None or (isinstance(value, str) and value == ""):
            return value
        return "<redacted>"
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): redact_config(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_config(item, key_name) for item in value]
    return value

@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def register_resolvers():
    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
        OmegaConf.register_new_resolver("int_div", lambda x, y: int(float(x) / float(y)))
        OmegaConf.register_new_resolver("not", lambda x: not x)
    except:
        pass # already registered



@dataclasses.dataclass
class GenerationsLogger:

    def log(self, loggers, samples, step, _type='val'):
        if 'wandb' in loggers:
            self.log_generations_to_wandb(samples, step, _type)
        if 'swanlab' in loggers:
            self.log_generations_to_swanlab(samples, step, _type)

    def log_generations_to_wandb(self, samples, step, _type='val'):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'table'):
            # Initialize the table on first call
            self.table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"{_type}/generations": new_table}, step=step)
        self.table = new_table

    def log_generations_to_swanlab(self, samples, step, _type='val'):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i+1}"))

        # Log to swanlab
        swanlab.log({f"{_type}/generations": swanlab_text_list}, step=step)
