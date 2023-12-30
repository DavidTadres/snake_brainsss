from dev import compare_zscore_data



rule check_zscore_rule:
    #shell:
    #    'python3 hello_world.py $args'
    threads: 8
    run:
        compare_zscore_data.run_comparison()