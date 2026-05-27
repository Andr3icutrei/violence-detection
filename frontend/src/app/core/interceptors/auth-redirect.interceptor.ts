import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

const AUTH_EXCLUDE_PATHS: string[] = [
  '/auth/login',
  '/auth/google-login',
  '/users/create',
  '/users/verify_account',
  '/users/resend_verification_email',
  '/users/verify_reset_password_token',
  '/users/request_reset_password',
  '/users/reset_password',
];

const shouldSkipRedirect = (url: string): boolean => AUTH_EXCLUDE_PATHS.some((path) => url.includes(path));

export const authRedirectInterceptor: HttpInterceptorFn = (req, next) => {
  const router = inject(Router);

  return next(req).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && err.status === 401 && !shouldSkipRedirect(req.url)) {
        const currentUrl = router.url;
        if (!currentUrl.startsWith('/portal/login')) {
          router.navigate(['/portal/login'], { queryParams: { returnUrl: currentUrl } });
        }
      }

      return throwError(() => err);
    }),
  );
};

