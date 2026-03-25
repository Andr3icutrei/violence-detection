from fastapi_mail import MessageSchema, MessageType, ConnectionConfig, FastMail
from fastapi import status, HTTPException

async def send_registration_email(to_email: str, verification_link: str, conf: ConnectionConfig) -> MessageSchema:

    mail_body = f"""
    <div style="font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif; background-color: #f4f7f6; padding: 40px 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">

            <h2 style="color: #2c3e50; text-align: center; margin-bottom: 20px;">Welcome to Violens!</h2>

            <p style="color: #555555; font-size: 16px; line-height: 1.6;">Hi there,</p>
            <p style="color: #555555; font-size: 16px; line-height: 1.6;">We're thrilled to have you on board. To get started and fully activate your account, please verify your email address by clicking the button below:</p>

            <div style="text-align: center; margin: 35px 0;">
                <a href="{verification_link}" style="background-color: #3498db; color: #ffffff; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: bold; display: inline-block;">Verify Email Address</a>
            </div>

            <p style="color: #7f8c8d; font-size: 14px; line-height: 1.5;">If you can't click the button, simply copy and paste this link into your browser's address bar:</p>
            <p style="color: #3498db; font-size: 14px; word-break: break-all; margin-bottom: 30px;">{verification_link}</p>

            <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 20px 0;">

            <p style="color: #95a5a6; font-size: 12px; text-align: center; line-height: 1.5;">If you did not create an account with us, please safely ignore this email.</p>
            <p style="color: #95a5a6; font-size: 12px; text-align: center;">Best regards,<br><strong style="color: #7f8c8d;">The Violens Team</strong></p>

        </div>
    </div>
    """

    message = MessageSchema(
        subject="Welcome to Violens! Please verify your email",
        recipients=[to_email],
        body=mail_body,
        subtype=MessageType.html
    )

    fm = FastMail(conf)

    try:
        await fm.send_message(message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while sending the email to {to_email}: {str(e)}"
        )

    return message

async def send_reset_password_email(to_email: str, reset_password_link: str, conf: ConnectionConfig) -> MessageSchema:

    mail_body = f"""
    <div style="font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif; background-color: #f4f7f6; padding: 40px 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">

            <h2 style="color: #2c3e50; text-align: center; margin-bottom: 20px;">Password Reset Request</h2>

            <p style="color: #555555; font-size: 16px; line-height: 1.6;">Hi there,</p>
            <p style="color: #555555; font-size: 16px; line-height: 1.6;">We received a request to reset your password. To proceed with resetting your password, please click the button below:</p>

            <div style="text-align: center; margin: 35px 0;">
                <a href="{reset_password_link}" style="background-color: #e74c3c; color: #ffffff; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-size: 16px; font-weight: bold; display: inline-block;">Reset Password</a>
            </div>

            <p style="color: #7f8c8d; font-size: 14px; line-height: 1.5;">If you can't click the button, simply copy and paste this link into your browser's address bar:</p>
            <p style="color: #e74c3c; font-size: 14px; word-break: break-all; margin-bottom: 30px;">{reset_password_link}</p>

            <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 20px 0;">

            <p style="color: #95a5a6; font-size: 12px; text-align: center; line-height: 1.5;">If you did not request a password reset, please safely ignore this email.</p>
            <p style="color: #95a5a6; font-size: 12px; text-align: center;">Best regards,<br><strong style="color: #7f8c8d;">The Violens Team</strong></p>
        </div>
    </div>
    """
    message = MessageSchema(
        subject="Password Reset Request for Your Violens Account",
        recipients=[to_email],
        body=mail_body,
        subtype=MessageType.html
    )

    fm = FastMail(conf)

    try:
        await fm.send_message(message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while sending the email to {to_email}: {str(e)}"
        )

    return message
