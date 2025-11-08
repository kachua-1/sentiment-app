import time

def fetch_google_play_reviews(app_id: str, lang='en', country='us', count=200):
    st.info(f"Fetching up to {count} reviews for {app_id}...")
    all_reviews, continuation_token = [], None
    fetched = 0
    progress = st.progress(0)
    total_batches = max(1, count // 200)
    start_time = time.time()

    while fetched < count:
        batch = min(200, count - fetched)
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                count=batch,
                continuation_token=continuation_token,
            )
        except Exception as e:
            st.warning(f"Stopped early due to error: {e}")
            break

        if not result:
            break
        all_reviews.extend(result)
        fetched += len(result)
        progress.progress(min(1.0, fetched / count))

        # stop if too slow (e.g. >90 seconds)
        if time.time() - start_time > 90:
            st.warning("⚠️ Fetching took too long — returning partial data.")
            break

        if continuation_token is None:
            break

    progress.empty()
    if not all_reviews:
        st.error("No reviews fetched — try a smaller count or different app ID.")
    else:
        st.success(f"Fetched {len(all_reviews)} reviews successfully!")
    return pd.DataFrame(all_reviews)
