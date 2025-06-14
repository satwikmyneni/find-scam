def preprocess(df):
    # Fill missing columns safely with empty strings
    df['title'] = df.get('title', '')
    df['company_profile'] = df.get('company_profile', '')
    df['description'] = df.get('description', '')
    df['requirements'] = df.get('requirements', '')
    df['benefits'] = df.get('benefits', '')

    # Combine relevant columns into a single text feature
    df["combined_text"] = (
        df["title"].fillna('') + ' ' +
        df["company_profile"].fillna('') + ' ' +
        df["description"].fillna('') + ' ' +
        df["requirements"].fillna('') + ' ' +
        df["benefits"].fillna('')
    )

    return df
