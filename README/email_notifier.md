# Experiment Email Notifier

Automatically sends email notifications when RAG optimization experiments finish

## Gmail App Password Setup

**Important**: You need an App Password, NOT your regular Gmail password.

1. **Enable 2-Step Verification**:
   - Go to [Google Account](https://myaccount.google.com/) → Security → 2-Step Verification → Enable

2. **Generate App Password**:
   - Go to Google Account → Security → **App passwords**
   - create a new app 
   - Copy the 16-character password (format: `xxxx xxxx xxxx xxxx`)
   - Use this as your `EMAIL_PASSWORD` (without spaces)

## Environment Variables

Create a `.env` file in your project root:

```env
EMAIL_SENDER=your.email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_RECIPIENTS=recipient1@email.com,recipient2@email.com  # optional, defaults to sender
```

## What You Get

The notifier automatically sends detailed HTML emails when your experiments complete, including:

- **Experiment Status**: Completed ✅, Failed ❌, or Early-Stopped ⚡
- **Performance Metrics**: Best scores, latency, trial counts
- **Best Configuration**: Optimal hyperparameters found
- **Inline Visualizations**: Optimization progress plots and Pareto fronts
- **Error Reporting**: Detailed error messages if experiments fail
- **Duration Tracking**: Total experiment runtime
