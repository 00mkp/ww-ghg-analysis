# importa all pipelines
import build_master
import clean
import eda
import regression
import classification
import clustering


# pipelines in order, each output feeds the next
PIPELINES = [
    ("build_master.py", build_master),
    ("clean.py", clean),
    ("eda.py", eda),
    ("regression.py", regression),
    ("classification.py", classification),
    ("clustering.py", clustering),
]


# Runs each pipeline's main() in order
def main():
    for name, module in PIPELINES:
        print(f"\n{'=' * 60}\nRunning {name}\n{'=' * 60}")
        module.main()  # call pipelines directly
    print(
        f"\n{'=' * 60}\nAll pipelines completed.\nReports written to reports/\n{'=' * 60}"
    )


if __name__ == "__main__":
    main()
