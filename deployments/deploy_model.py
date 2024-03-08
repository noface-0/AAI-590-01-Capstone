import threading
from fastapi import FastAPI

from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER
from config.models import ERL_PARAMS, SAC_PARAMS
from config.training import TIME_INTERVAL, AGENT
from environments.alpaca import AlpacaPaperTrading
from utils.utils import get_var
from deployments.s3_utils import save_model_from_s3

app = FastAPI()

action_dim = len(DOW_30_TICKER)
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim

API_KEY = get_var("ALPACA_API_KEY")
API_SECRET = get_var("ALPACA_API_SECRET")
API_BASE_URL = get_var("ALPACA_API_BASE_URL")


def start_trading():
    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": SAC_PARAMS
    }
    bucket_name = 'rl-trading-v1-runs'
    model_s3_path = 'runs/models/actor.pth'

    # retrieve most recent model
    try:
        save_model_from_s3(
            bucket_name=bucket_name,
            s3_path=model_s3_path,
            local_file_path='models/runs/papertrading_erl_retrain'
        )
        print("loaded model from S3")
    except Exception:
        pass

    params = agent_configs.get(AGENT)

    paper_trading_erl = AlpacaPaperTrading(
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL,
        drl_lib='elegantrl',
        agent=AGENT,
        cwd='models/runs/papertrading_erl_retrain',
        net_dim=params['net_dimension'],
        state_dim=state_dim,
        action_dim=action_dim,
        API_KEY=API_KEY,
        API_SECRET=API_SECRET,
        API_BASE_URL=API_BASE_URL,
        tech_indicator_list=INDICATORS,
        turbulence_thresh=30,
        max_stock=1e2
    )
    paper_trading_erl.run()


@app.on_event("startup")
def on_startup():
    thread = threading.Thread(target=start_trading)
    thread.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)