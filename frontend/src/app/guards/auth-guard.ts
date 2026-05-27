import { CanActivateFn, Router } from '@angular/router';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth/auth-service';
import { UserResponseDto } from '../core/api/models/user-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';
import { of } from 'rxjs';

export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  return authService.me().pipe(
    map((data: UserResponseDto) => {
      const adminRole: boolean = !!data.is_admin;
      if (route.data['role'] === 'admin') {
        if (adminRole) {
          return true;
        } else {
          router.navigate(['/dashboard'], { queryParams: { returnUrl: state.url } });
          return false;
        }
      }
      return true;
    }),
    catchError((err: HttpErrorResponse) => {
      router.navigate(['/portal/login'], { queryParams: { returnUrl: state.url } });
      return of(false);
    }),
  );
};
