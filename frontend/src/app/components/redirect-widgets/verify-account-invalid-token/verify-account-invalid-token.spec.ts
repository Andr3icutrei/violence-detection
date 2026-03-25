import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountInvalidToken } from './verify-account-invalid-token';

describe('VerifyAccountInvalidToken', () => {
  let component: VerifyAccountInvalidToken;
  let fixture: ComponentFixture<VerifyAccountInvalidToken>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountInvalidToken],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountInvalidToken);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
