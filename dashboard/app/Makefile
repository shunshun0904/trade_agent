# sam build が呼ぶ（template.yaml の BuildMethod: makefile）。CloudShell（x86_64）から arm64 の Lambda 用に組む。
# numpy と pandas は aarch64 の wheel を pip で取り、研究用の bbdata / bbresearch とモデル（JSON）をそのまま同梱する。
build-DashboardFunction:
	cp *.py $(ARTIFACTS_DIR)/
	cp page.html $(ARTIFACTS_DIR)/
	rm -rf $(ARTIFACTS_DIR)/bbdata $(ARTIFACTS_DIR)/bbresearch
	cp -r ../../bbdata ../../bbresearch $(ARTIFACTS_DIR)/
	find $(ARTIFACTS_DIR)/bbdata $(ARTIFACTS_DIR)/bbresearch -name __pycache__ -type d -prune -exec rm -rf {} +
	test -f ../../reports/direction_deploy/model_B.json && cp ../../reports/direction_deploy/model_B.json $(ARTIFACTS_DIR)/ || echo "model_B.json がない（確率は出ない）"
	python3 -m pip install --quiet --target $(ARTIFACTS_DIR) --platform manylinux2014_aarch64 --implementation cp \
	    --python-version 3.12 --only-binary=:all: --upgrade "numpy==2.4.*" "pandas==3.0.*"
	find $(ARTIFACTS_DIR) -name "*.dist-info" -type d -prune -exec rm -rf {} +
	find $(ARTIFACTS_DIR) -path "*/tests" -type d -prune -exec rm -rf {} +
