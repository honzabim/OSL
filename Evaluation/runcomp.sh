for i in `seq 1 100`;
do
        julia evaluation_svae_vs_m1.jl isolet $i &
        julia evaluation_svae_vs_m1.jl letter-recognition $i &
        julia evaluation_svae_vs_m1.jl libras $i &
        julia evaluation_svae_vs_m1.jl multiple-features $i &
        julia evaluation_svae_vs_m1.jl pendigits $i &
        julia evaluation_svae_vs_m1.jl yeast $i &
        julia evaluation_svae_vs_m1.jl cardiotocography $i &
        julia evaluation_svae_vs_m1.jl ecoli $i &
        julia evaluation_svae_vs_m1.jl page-blocks $i &
        julia evaluation_svae_vs_m1.jl statlog-satimage $i &
        julia evaluation_svae_vs_m1.jl statlog-segment $i &
        julia evaluation_svae_vs_m1.jl statlog-shuttle $i &
        julia evaluation_svae_vs_m1.jl statlog-vehicle $i &
        julia evaluation_svae_vs_m1.jl breast-tissue $i &
        julia evaluation_svae_vs_m1.jl synthetic-control-chart $i &
        julia evaluation_svae_vs_m1.jl wall-following-robot $i &
        julia evaluation_svae_vs_m1.jl waveform-1 $i &
        julia evaluation_svae_vs_m1.jl waveform-2 $i &
        julia evaluation_svae_vs_m1.jl wine $i &
        julia evaluation_svae_vs_m1.jl iris $i &

        julia evaluation_svae_vs_m1.jl abalone $i &
        julia evaluation_svae_vs_m1.jl blood-transfusion $i &
        julia evaluation_svae_vs_m1.jl breast-cancer-wisconsin $i &
        julia evaluation_svae_vs_m1.jl gisette $i &
        julia evaluation_svae_vs_m1.jl glass $i &
        julia evaluation_svae_vs_m1.jl haberman $i &
        julia evaluation_svae_vs_m1.jl ionosphere $i &
        julia evaluation_svae_vs_m1.jl madelon $i &
        julia evaluation_svae_vs_m1.jl magic-telescope $i &
        julia evaluation_svae_vs_m1.jl miniboone $i &
        julia evaluation_svae_vs_m1.jl parkinsons $i &
        julia evaluation_svae_vs_m1.jl pima-indians $i &
        julia evaluation_svae_vs_m1.jl sonar $i &
        julia evaluation_svae_vs_m1.jl spect-heart $i &
        julia evaluation_svae_vs_m1.jl vertebral-column $i &
        julia evaluation_svae_vs_m1.jl musk-2 $i
done
