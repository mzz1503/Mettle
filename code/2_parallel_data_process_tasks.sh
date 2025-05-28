TOTAL_TASKS=50
for i in $(seq 0 $((TOTAL_TASKS - 1)))
do
   python generate_candidates.py --index_task $i --num_task $TOTAL_TASKS &
done
wait
for i in $(seq 0 $((TOTAL_TASKS - 1)))
do
   python preprocess_candidates.py --index_task $i --num_task $TOTAL_TASKS &
done
wait
echo "All tasks are done!"
