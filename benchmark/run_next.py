"""Run the next planned experiment, preserving logs and validation comparison."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark.run_learning_rates import execute, read_csv
from benchmark_log import ROOT, write_csv


def main():
    plan_path = ROOT / 'benchmark/plan.csv'
    plan = read_csv(plan_path)
    item = next((row for row in plan if row['status'] == 'planned'), None)
    if item is None:
        print('No planned experiments remain.')
        return
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    name = f"{item['experiment_id']}_{stamp}"
    suite = ROOT / 'benchmark/suites' / name
    run = ROOT / 'benchmark/runs' / name
    suite.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-X', 'utf8', '-u', 'train_step.py', '--train', '--run-dir', str(run)]
    for field in ('optimizer', 'lr', 'weight_decay', 'batch_size', 'image_size', 'seed', 'epochs'):
        command.extend(['--' + field.replace('_', '-'), item[field]])
    manifest = {'status': 'running', 'experiment': dict(item), 'command': command,
                'run_dir': str(run.relative_to(ROOT)), 'selection': 'validation only; no test evaluation'}
    def save(status):
        manifest['status'] = status
        (suite / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        item['status'] = status
        item['notes'] = f"{manifest['experiment']['notes']}; artifacts: {suite.relative_to(ROOT).as_posix()}"
        write_csv(plan_path, list(plan[0]), plan)
    save('running')
    (suite / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    try:
        execute(command, suite / 'train.log')
    except BaseException:
        save('failed_or_interrupted')
        raise
    row = read_csv(run / 'result.csv')[0]
    write_csv(suite / 'results.csv', list(row), [row])
    save('completed')
    print(f'COMPLETE: {suite}', flush=True)


if __name__ == '__main__':
    main()
