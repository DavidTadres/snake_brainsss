from dev import compare_zscore_data



rule check_zscore_rule:
    #shell:
    #    'python3 hello_world.py $args'
    threads: 8
    resources: mem_mb='100G'
    run:
        compare_zscore_data.run_comparison()